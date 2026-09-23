import argparse
import base64
import getpass
import hashlib
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TextIO

from . import _project
from ._client import Client
from ._config import DEFAULT_BASE_URL, ENV_API_KEY
from ._exceptions import APIError

ENV_AGENT_ID = "ASSEMBLYAI_AGENT_ID"
ENV_BASE_URL = "ASSEMBLYAI_BASE_URL"

PROG = "assemblyai-agents"
DEPLOYMENTS_PATH = "/v1/agent-deployments"
SECRETS_PATH = "/v1/tool-secrets"
AGENTS_PATH = "/v1/agents"
MAX_SOURCE_CHARACTERS = 262144

# How many excluded entries are named before the rest are counted. A pruned
# directory is one entry, not one per file inside it, so a real project's list
# is short and this only bites on a tree that was never meant to be deployed.
MAX_EXCLUSIONS_SHOWN = 12

STATUS_READY = "ready"

# Statuses are plain strings, never the generated `DeploymentStatus` enum: that
# enum raises on a value it has never heard of, which would abort a deploy that
# is running fine against a newer service.
FAILURE_EXPLANATIONS = {
    "import_failed": "Your tool module did not import.",
    "no_tools_found": (
        "The module imported but defined no tools. Every tool is a function "
        "decorated with @tool(), and at least one has to be there."
    ),
    "dependencies_failed": (
        "Your module's dependencies could not be installed. The resolver's own "
        "output is below."
    ),
    "timed_out": (
        "The deployment did not finish in time. Nothing changed on your agent."
    ),
    "internal_error": (
        "The deployment failed inside AssemblyAI rather than in your code. "
        "Quote the deployment ID below to support."
    ),
}

# The statuses the service treats as not yet settled. A status this version has
# never heard of is treated as unsettled too, so a newer service can add one
# without this command calling a running deployment a failure.
PENDING_STATUSES = frozenset({"pending", "building"})

FIRST_POLL_SECONDS = 1.0
MAX_POLL_SECONDS = 5.0
POLL_BACKOFF = 1.5
HEARTBEAT_SECONDS = 30.0
# Longer than the service's own ceiling on a pending deployment, so a stuck
# deploy is reported by the API as timed_out rather than abandoned here.
WAIT_LIMIT_SECONDS = 20 * 60

_TRANSIENT_LINE_WIDTH = 72

# What a shell reports for a command stopped with Ctrl-C.
INTERRUPTED_EXIT_CODE = 130
# `deployments status` only: the deployment has neither succeeded nor failed, so
# a build job can tell "not finished" from "finished badly" on the exit status.
PENDING_EXIT_CODE = 3

# Mirrors the pattern the API enforces on the secret name. It is a path
# component there, so a rejection arrives as a complaint about a URL segment;
# checking here turns a hyphen in a name into a sentence about the name.
SECRET_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
MAX_SECRET_VALUE_LENGTH = 8192
# The API refuses the 101st secret. Named here only so the local message can
# point at `secrets list`; the server's own refusal is what gets printed.
MAX_SECRETS_PER_ACCOUNT = 100

DEFAULT_DEPLOYMENTS_LIMIT = 50
MAX_DEPLOYMENTS_LIMIT = 200


class UsageError(Exception):
    pass


def _format_elapsed(seconds: float) -> str:
    whole = int(seconds)
    if whole < 60:
        return f"{whole}s"
    return f"{whole // 60}m {whole % 60:02d}s"


class _Progress:
    def __init__(self, stream: TextIO, interactive: bool) -> None:
        self._stream = stream
        self._interactive = interactive
        self._status: Optional[str] = None
        self._transient = False
        self._last_line_at = 0.0

    def update(self, status: str, elapsed: float) -> None:
        if status != self._status:
            self._status = status
            self._commit(f"  {status} ({_format_elapsed(elapsed)})", elapsed)
        elif self._interactive:
            self._write_transient(f"  {status} ({_format_elapsed(elapsed)})")
        elif elapsed - self._last_line_at >= HEARTBEAT_SECONDS:
            self._commit(f"  still {status} ({_format_elapsed(elapsed)})", elapsed)

    def finish(self) -> None:
        self._clear()

    def _commit(self, text: str, elapsed: float) -> None:
        self._clear()
        self._stream.write(text + "\n")
        self._stream.flush()
        self._last_line_at = elapsed

    def _write_transient(self, text: str) -> None:
        self._stream.write("\r" + text.ljust(_TRANSIENT_LINE_WIDTH))
        self._stream.flush()
        self._transient = True

    def _clear(self) -> None:
        if not self._transient:
            return
        self._stream.write("\r" + " " * _TRANSIENT_LINE_WIDTH + "\r")
        self._stream.flush()
        self._transient = False


def read_source(path: str) -> str:
    try:
        source = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        raise UsageError(f"No such file: {path}")
    except IsADirectoryError:
        raise UsageError(f"{path} is a directory that cannot be opened.")
    except UnicodeDecodeError:
        raise UsageError(f"{path} is not UTF-8 text, so it is not a Python module.")
    except OSError as exc:
        raise UsageError(f"Cannot read {path}: {exc.strerror}.")
    if not source.strip():
        raise UsageError(f"{path} is empty; there is nothing to deploy.")
    if len(source) > MAX_SOURCE_CHARACTERS:
        raise UsageError(
            f"{path} is {len(source):,} characters. A tool module can be at most "
            f"{MAX_SOURCE_CHARACTERS:,}."
        )
    return source


def read_upload(path: str, out: TextIO) -> Dict[str, str]:
    """The request body's code field, chosen by what PATH turned out to be.

    A directory is packed and sent whole; a single file is sent as it always
    was. The service treats the second as shorthand for the first, so both
    spellings of one project deduplicate to one stored project and one image.
    """
    target = Path(path)
    if not target.is_dir():
        return {"source": read_source(path)}
    try:
        project = _project.read_project(target)
        archive = _project.build_archive(project.files)
    except _project.ProjectError as exc:
        raise UsageError(str(exc))
    _print_packing(path, project, archive, out)
    return {"archive": base64.b64encode(archive).decode("ascii")}


def _print_packing(
    path: str, project: _project.Project, archive: bytes, out: TextIO
) -> None:
    digest = hashlib.sha256(archive).hexdigest()
    print(
        f"Packed {len(project.files)} files from {path}, {len(archive):,} bytes "
        f"(sha256 {digest[:12]}).",
        file=out,
    )
    if project.excluded:
        print("Not uploaded:", file=out)
        for name, reason in project.excluded[:MAX_EXCLUSIONS_SHOWN]:
            print(f"  {name} — {reason}", file=out)
        remaining = len(project.excluded) - MAX_EXCLUSIONS_SHOWN
        if remaining > 0:
            print(f"  and {remaining} more", file=out)
        if any(
            reason == _project.ENV_FILE_REASON for _, reason in project.excluded
        ):
            print(
                f"  Credentials belong in {PROG} secrets set NAME, which your "
                f'tools read back with ctx.secret("NAME").',
                file=out,
            )
    for name in project.empty_directories:
        # A deployment is a set of file paths, so there is nothing to carry an
        # empty directory. Said here because the symptom otherwise arrives as an
        # ImportError from a package that looks present on the customer's disk.
        print(
            f"  {name}/ holds no files, so it will not exist in the deployment.",
            file=out,
        )


def _print_api_error(exc: APIError, err: TextIO) -> None:
    print(f"AssemblyAI returned HTTP {exc.status}.", file=err)
    print(exc.message, file=err)
    if exc.code:
        print(f"Error code: {exc.code}", file=err)
    if exc.request_id:
        print(f"Request ID: {exc.request_id}", file=err)


def _print_failure(
    status: str,
    detail: Optional[str],
    deployment_id: str,
    elapsed: float,
    err: TextIO,
) -> None:
    explanation = FAILURE_EXPLANATIONS.get(status)
    if explanation is None:
        print(
            f"Gave up waiting after {_format_elapsed(elapsed)}. The deployment is "
            f"still {status} and is still running at AssemblyAI.",
            file=err,
        )
        print(
            f"Ask again with: {PROG} deployments status {deployment_id}",
            file=err,
        )
    else:
        print(f"Deploy failed after {_format_elapsed(elapsed)}: {status}", file=err)
        print(explanation, file=err)
    if detail:
        print("", file=err)
        print("AssemblyAI reported:", file=err)
        print("", file=err)
        print(detail, file=err)
    print("", file=err)
    print(f"Deployment {deployment_id}", file=err)


def _wait_for_deployment(
    client: Client,
    deployment_id: str,
    *,
    progress: _Progress,
    sleep: Callable[[float], None],
    monotonic: Callable[[], float],
) -> tuple:
    started = monotonic()
    interval = FIRST_POLL_SECONDS
    while True:
        sleep(interval)
        interval = min(interval * POLL_BACKOFF, MAX_POLL_SECONDS)
        body = client.request("GET", f"{DEPLOYMENTS_PATH}/{deployment_id}")
        status = str(body.get("status"))
        elapsed = monotonic() - started
        progress.update(status, elapsed)
        if status == STATUS_READY or status in FAILURE_EXPLANATIONS:
            return body, elapsed
        if elapsed >= WAIT_LIMIT_SECONDS:
            return body, elapsed


def _print_create_hint(exc: APIError, base_url: str, err: TextIO) -> None:
    """Add what the server cannot know: which host we asked, and where to look."""
    if exc.status == 404:
        print("", file=err)
        print(
            f"That agent was not found at {base_url}. Agents are stored per host, "
            f"so one created against a different base URL is invisible here. "
            f"Check --base-url (or {ENV_BASE_URL}).",
            file=err,
        )
    elif exc.status == 429:
        print("", file=err)
        print(
            "That is a cap on deployments running at once, not on how many you "
            "may have. An earlier one finishing frees a slot.",
            file=err,
        )
        print(f"See what is running: {PROG} deployments list", file=err)


def deploy(
    client: Client,
    *,
    path: str,
    upload: Dict[str, str],
    agent_id: str,
    out: TextIO,
    err: TextIO,
    sleep: Callable[[float], None],
    monotonic: Callable[[], float],
    interactive: bool,
    base_url: str = DEFAULT_BASE_URL,
) -> int:
    print(f"Deploying {path} to agent {agent_id}.", file=out)
    try:
        created: Any = client.request(
            "POST",
            DEPLOYMENTS_PATH,
            json={"agent_id": agent_id, **upload},
            idempotent=True,
        )
    except APIError as exc:
        _print_api_error(exc, err)
        _print_create_hint(exc, base_url, err)
        return 1

    deployment_id = str(created.get("id"))
    print(f"Created deployment {deployment_id}.", file=out)

    progress = _Progress(out, interactive)
    progress.update(str(created.get("status")), 0.0)
    try:
        body, elapsed = _wait_for_deployment(
            client,
            deployment_id,
            progress=progress,
            sleep=sleep,
            monotonic=monotonic,
        )
    except APIError as exc:
        progress.finish()
        _print_api_error(exc, err)
        print(
            f"Deployment {deployment_id} was not cancelled and may still finish.",
            file=err,
        )
        print(f"Ask again with: {PROG} deployments status {deployment_id}", file=err)
        return 1
    except KeyboardInterrupt:
        progress.finish()
        print(
            f"Stopped waiting. Deployment {deployment_id} was not cancelled and "
            f"is still running at AssemblyAI.",
            file=err,
        )
        print(f"Ask again with: {PROG} deployments status {deployment_id}", file=err)
        return INTERRUPTED_EXIT_CODE
    progress.finish()

    status = str(body.get("status"))
    if status == STATUS_READY:
        print(f"Deployed in {_format_elapsed(elapsed)}.", file=out)
        print(
            f"AssemblyAI is now running the tools in {path} for agent {agent_id}.",
            file=out,
        )
        return 0

    _print_failure(status, body.get("detail"), deployment_id, elapsed, err)
    return 1


def check_secret_name(name: str) -> None:
    if SECRET_NAME_PATTERN.match(name):
        return
    if not name:
        raise UsageError("The secret name is empty.")
    if len(name) > 64:
        raise UsageError(
            f"Secret name {name!r} is {len(name)} characters. A name can be at "
            f"most 64."
        )
    raise UsageError(
        f"Secret name {name!r} is not usable. A name starts with a letter or "
        f"underscore and then uses only letters, digits and underscores — no "
        f"hyphens, dots or spaces. It is also the name you read it back by, as "
        f'ctx.secret("{name}").'
    )


def read_secret_value(name: str, stdin: TextIO, interactive: bool) -> str:
    """Take the value from the terminal or from standard input, never an option.

    An option would be recorded in the shell history and visible in the process
    list to every other user on the machine.
    """
    if interactive:
        value = getpass.getpass(f"Value for {name} (not echoed): ")
    else:
        value = stdin.read()
        # One trailing newline only, so `echo v |` and a heredoc both send `v`
        # while a value that genuinely ends in a blank line keeps it.
        if value.endswith("\n"):
            value = value[:-1]
    if not value:
        raise UsageError(
            f"No value given for {name}. The value is read from standard input, "
            f"never from an option, because options are saved in your shell "
            f"history."
        )
    if len(value) > MAX_SECRET_VALUE_LENGTH:
        raise UsageError(
            f"That value is {len(value):,} characters. A secret can be at most "
            f"{MAX_SECRET_VALUE_LENGTH:,}."
        )
    return value


def secrets_set(
    client: Client,
    *,
    name: str,
    value: str,
    out: TextIO,
    err: TextIO,
) -> int:
    try:
        body: Any = client.request(
            "PUT", f"{SECRETS_PATH}/{name}", json={"value": value}
        )
    except APIError as exc:
        _print_api_error(exc, err)
        if exc.code == "tool_secret_limit_exceeded":
            print("", file=err)
            print(f"See what you have: {PROG} secrets list", file=err)
        return 1
    print(
        f"Set {body.get('name', name)} (updated {body.get('updated_at')}). "
        f'Read it in a tool with ctx.secret("{name}").',
        file=out,
    )
    return 0


def _column_widths(rows: list, headers: tuple) -> list:
    widths = [len(h) for h in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))
    return widths


def _print_table(
    rows: list,
    headers: tuple,
    out: TextIO,
    show_header: bool,
) -> None:
    widths = _column_widths(rows, headers)
    if show_header:
        print("  ".join(h.ljust(w) for h, w in zip(headers, widths)).rstrip(), file=out)
    for row in rows:
        print(
            "  ".join(c.ljust(w) for c, w in zip(row, widths)).rstrip(),
            file=out,
        )


def secrets_list(
    client: Client,
    *,
    out: TextIO,
    err: TextIO,
    interactive: bool,
) -> int:
    try:
        body: Any = client.request("GET", SECRETS_PATH)
    except APIError as exc:
        _print_api_error(exc, err)
        return 1
    secrets = body.get("secrets") or []
    if not secrets:
        print(
            f"No secrets on this account. Add one with: {PROG} secrets set NAME",
            file=out,
        )
        return 0
    rows = [
        [
            str(s.get("name", "")),
            str(s.get("created_at", "")),
            str(s.get("updated_at", "")),
        ]
        for s in secrets
    ]
    # The header is for a human reading a terminal. Piping into awk or cut is the
    # other use, and a header would become a bogus first record there.
    _print_table(rows, ("NAME", "CREATED", "UPDATED"), out, interactive)
    return 0


def secrets_delete(
    client: Client,
    *,
    name: str,
    out: TextIO,
    err: TextIO,
) -> int:
    try:
        client.request("DELETE", f"{SECRETS_PATH}/{name}")
    except APIError as exc:
        if exc.status == 404:
            print(f"There is no secret named {name} on this account.", file=err)
            return 1
        _print_api_error(exc, err)
        return 1
    print(f"Deleted {name}.", file=out)
    return 0


def _live_deployment_ids(client: Client, agent_id: str) -> set:
    """Which of this agent's tools came from which deployment.

    A tool carrying a deployment id is one AssemblyAI runs; the agent record is
    the only place that mapping exists, so this is the only way to show a
    customer which deployment is actually serving.
    """
    try:
        agent: Any = client.request("GET", f"{AGENTS_PATH}/{agent_id}")
    except APIError:
        # The listing is still worth printing without the marker.
        return set()
    live = set()
    for tool in agent.get("tools") or []:
        if isinstance(tool, dict) and tool.get("deployment_id"):
            live.add(str(tool["deployment_id"]))
    return live


def deployments_list(
    client: Client,
    *,
    agent_id: Optional[str],
    limit: Optional[int],
    fetch_all: bool,
    out: TextIO,
    err: TextIO,
    interactive: bool,
) -> int:
    params: dict = {}
    if agent_id:
        params["agent_id"] = agent_id
    if limit is not None:
        params["limit"] = limit

    has_more = False
    try:
        if fetch_all:
            deployments = list(
                client.paginate(
                    DEPLOYMENTS_PATH, item_key="agent_deployments", params=params
                )
            )
        else:
            body: Any = client.request("GET", DEPLOYMENTS_PATH, params=params)
            deployments = body.get("agent_deployments") or []
            has_more = bool(body.get("has_more"))
    except APIError as exc:
        _print_api_error(exc, err)
        return 1

    if not deployments:
        if agent_id:
            print(f"No deployments for agent {agent_id}.", file=out)
        else:
            print("No deployments on this account.", file=out)
        return 0

    live = _live_deployment_ids(client, agent_id) if agent_id else set()

    headers = ("DEPLOYMENT", "AGENT", "STATUS", "CREATED")
    rows = []
    for d in deployments:
        status = str(d.get("status", ""))
        if str(d.get("id")) in live:
            status = f"{status} (serving)"
        rows.append(
            [
                str(d.get("id", "")),
                str(d.get("agent_id", "")),
                status,
                str(d.get("created_at", "")),
            ]
        )
    _print_table(rows, headers, out, interactive)

    if has_more:
        print("", file=out)
        print("There are more. Add --all to list every one.", file=out)
    return 0


def deployments_status(
    client: Client,
    *,
    deployment_id: str,
    out: TextIO,
    err: TextIO,
) -> int:
    """Read one deployment, which is also what resolves a stale pending row.

    A listing reports the stored status without the age check this read applies,
    and nothing sweeps the table, so a deployment whose builder died reads
    `pending` in a listing indefinitely. This is the call that settles it.
    """
    try:
        body: Any = client.request("GET", f"{DEPLOYMENTS_PATH}/{deployment_id}")
    except APIError as exc:
        _print_api_error(exc, err)
        return 1

    status = str(body.get("status"))
    print(f"Deployment {body.get('id', deployment_id)}", file=out)
    print(f"  agent    {body.get('agent_id')}", file=out)
    print(f"  status   {status}", file=out)
    print(f"  created  {body.get('created_at')}", file=out)
    print(f"  updated  {body.get('updated_at')}", file=out)

    detail = body.get("detail")
    if detail:
        print("", file=out)
        print("AssemblyAI reported:", file=out)
        print("", file=out)
        print(detail, file=out)

    if status == STATUS_READY:
        return 0
    if status in PENDING_STATUSES:
        return PENDING_EXIT_CODE
    explanation = FAILURE_EXPLANATIONS.get(status)
    if explanation:
        print("", file=out)
        print(explanation, file=out)
        return 1
    # A status this version has never heard of is not called a failure; a newer
    # service may have added another in-progress state.
    return PENDING_EXIT_CODE


def deployments_delete(
    client: Client,
    *,
    deployment_id: str,
    out: TextIO,
    err: TextIO,
) -> int:
    try:
        client.request("DELETE", f"{DEPLOYMENTS_PATH}/{deployment_id}")
    except APIError as exc:
        if exc.status == 404:
            print(f"There is no deployment {deployment_id} on this account.", file=err)
            return 1
        # The server refuses for two different reasons and says which, naming the
        # agent that still holds it. Reprinting that in our own words would lose
        # the agent id, so it goes through unaltered.
        _print_api_error(exc, err)
        return 1
    print(f"Deleted deployment {deployment_id}.", file=out)
    return 0


def _base_url_parser() -> argparse.ArgumentParser:
    """Carried by every subcommand via `parents=`, so it may be typed last.

    A top-level-only option would force `assemblyai-agents --base-url URL secrets
    set NAME`, which nobody guesses.
    """
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument(
        "--base-url",
        default=os.environ.get(ENV_BASE_URL, DEFAULT_BASE_URL),
        metavar="URL",
        help=(
            f"API to talk to, for testing against a non-production endpoint "
            f"(default: {DEFAULT_BASE_URL}, or set {ENV_BASE_URL})."
        ),
    )
    return parent


def _positive_limit(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a whole number.")
    if not 1 <= value <= MAX_DEPLOYMENTS_LIMIT:
        raise argparse.ArgumentTypeError(
            f"--limit is between 1 and {MAX_DEPLOYMENTS_LIMIT}; got {value}."
        )
    return value


def build_parser() -> argparse.ArgumentParser:
    common = _base_url_parser()
    parser = argparse.ArgumentParser(
        prog=PROG,
        description=(
            "Deploy tool code for AssemblyAI to run (managed tools). Tools your "
            "own process or your own server answers are configured in Python, "
            "not here."
        ),
        epilog=(
            f"Your API key is read from the {ENV_API_KEY} environment variable. "
            f"It is never accepted as an option, because options are saved in "
            f"your shell history."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    deploy_parser = subparsers.add_parser(
        "deploy",
        parents=[common],
        help="Upload your tool code to an agent and wait for it to go live.",
        description=(
            f"Uploads PATH to AssemblyAI and waits for the deployment to "
            f"finish. AssemblyAI runs the tools in it. PATH is either one "
            f"Python file, or a project directory with {_project.ENTRY_NAME} at "
            f"the top of it — your tools are read from that file, and every "
            f"other module in the directory is importable from it. The code has "
            f"to import cleanly and define at least one function decorated with "
            f"@tool(). If it fails to import, the error from your own code is "
            f"printed. Deploying replaces the tools AssemblyAI runs for this "
            f"agent and leaves tools your own server answers alone. Exits 0 only "
            f"when the agent is serving the new tools, so a build job can depend "
            f"on the exit status."
        ),
        epilog=(
            f"A directory is uploaded whole, minus these: every .env and .env.* "
            f"file, because credentials belong in `{PROG} secrets set`; "
            f"__pycache__, .git, .pyc and .pyo, which AssemblyAI drops anyway; "
            f"editor and tool caches; any directory holding a "
            f"{_project.VENV_MARKER}, which makes it a virtual environment; and "
            f"anything matched by a {_project.IGNORE_FILE} file at the top of "
            f"the project, one glob per line. Everything else travels, "
            f"including pyproject.toml, requirements.txt and lock files, and "
            f"every exclusion is printed. A symbolic link, a file that is not "
            f"UTF-8 text, and a project over "
            f"{_project.MAX_ARCHIVE_BYTES // (1024 * 1024)} MiB or "
            f"{_project.MAX_FILES} files are refused rather than quietly "
            f"dropped."
        ),
    )
    deploy_parser.add_argument(
        "path",
        metavar="PATH",
        help=(
            f"Python file defining your tools, or a project directory with "
            f"{_project.ENTRY_NAME} at the top of it."
        ),
    )
    deploy_parser.add_argument(
        "--agent",
        default=os.environ.get(ENV_AGENT_ID),
        metavar="AGENT_ID",
        help=(
            f"Agent to deploy to, for example "
            f"agent_b4c9e0d27a314c6e9f5a8d2e6c1b0a47 (or set {ENV_AGENT_ID})."
        ),
    )

    secrets_parser = subparsers.add_parser(
        "secrets",
        help="Store the credentials your deployed tools read.",
        description=(
            "Credentials held by AssemblyAI and handed to your deployed tools. A "
            'tool reads one with ctx.secret("NAME"). Values are read fresh for '
            "each session, so a changed secret takes effect on the next call "
            "without redeploying. A value is never accepted as an argument."
        ),
    )
    secrets_sub = secrets_parser.add_subparsers(dest="subcommand", required=True)

    secrets_set_parser = secrets_sub.add_parser(
        "set",
        parents=[common],
        help="Store or replace one secret, reading the value from stdin.",
        description=(
            "Stores NAME, taking the value from standard input and nowhere else "
            "— at a terminal it is prompted for and not echoed, otherwise it is "
            "read from the pipe. It is never an argument, because arguments are "
            "saved in your shell history and visible in the process list. "
            "Setting a name that already exists replaces its value. The name is "
            'the one your tool reads back with ctx.secret("NAME").'
        ),
    )
    secrets_set_parser.add_argument("name", metavar="NAME", help="Name of the secret.")

    secrets_list_parser = secrets_sub.add_parser(
        "list",
        parents=[common],
        help="List the secrets on this account, by name.",
        description=(
            "Lists every secret on the account by name, with when it was created "
            "and last changed. Values are never returned by the API. The header "
            "is printed only to a terminal, so piping this into another command "
            "needs no stripping."
        ),
    )

    secrets_delete_parser = secrets_sub.add_parser(
        "delete",
        parents=[common],
        help="Delete one secret.",
        description=(
            "Deletes NAME. There is no confirmation prompt: the value cannot be "
            "recovered but can be set again, and a prompt would break every "
            "script that pipes into this command."
        ),
    )
    secrets_delete_parser.add_argument(
        "name", metavar="NAME", help="Name of the secret."
    )

    deployments_parser = subparsers.add_parser(
        "deployments",
        help="Inspect and remove the tool code AssemblyAI is running.",
        description=(
            "Each deploy produces a deployment. These commands show which ones "
            "exist, which one an agent is serving, and remove the ones you no "
            "longer need."
        ),
    )
    deployments_sub = deployments_parser.add_subparsers(
        dest="subcommand", required=True
    )

    deployments_list_parser = deployments_sub.add_parser(
        "list",
        parents=[common],
        help="List deployments, newest first.",
        description=(
            "Lists deployments newest first. With --agent, the deployment that "
            "agent is actually serving is marked. A status here is the stored "
            "one and is not re-checked for age, so a deployment whose build died "
            f"can keep reading pending; `{PROG} deployments status ID` is the "
            "read that settles it."
        ),
    )
    deployments_list_parser.add_argument(
        "--agent",
        default=None,
        metavar="AGENT_ID",
        help="Only this agent's deployments, and mark the one it is serving.",
    )
    deployments_list_parser.add_argument(
        "--limit",
        type=_positive_limit,
        default=None,
        metavar="N",
        help=(
            f"How many to return, 1 to {MAX_DEPLOYMENTS_LIMIT} "
            f"(default {DEFAULT_DEPLOYMENTS_LIMIT})."
        ),
    )
    deployments_list_parser.add_argument(
        "--all",
        action="store_true",
        dest="fetch_all",
        help="Follow every page instead of stopping at the first.",
    )

    deployments_status_parser = deployments_sub.add_parser(
        "status",
        parents=[common],
        help="Read one deployment and exit 0 ready, 1 failed, 3 still running.",
        description=(
            "Reads one deployment and reports it on the exit status: 0 when it "
            "is ready, 1 when it failed, 3 when it is still running. Unlike a "
            "listing, this read re-checks a pending deployment's age, so it is "
            "what resolves one whose build died. Use it after a deploy you "
            "stopped waiting for."
        ),
    )
    deployments_status_parser.add_argument(
        "deployment_id", metavar="DEPLOYMENT_ID", help="Deployment to read."
    )

    deployments_delete_parser = deployments_sub.add_parser(
        "delete",
        parents=[common],
        help="Delete one deployment.",
        description=(
            "Deletes DEPLOYMENT_ID. AssemblyAI refuses while the deployment is "
            "still building, and while it is still providing an agent's tools; "
            "in both cases it says so and what to do instead."
        ),
    )
    deployments_delete_parser.add_argument(
        "deployment_id", metavar="DEPLOYMENT_ID", help="Deployment to delete."
    )

    return parser


def _require_api_key() -> None:
    if not os.environ.get(ENV_API_KEY):
        raise UsageError(
            f"No API key. Set the {ENV_API_KEY} environment variable; it is "
            f"never read from an option, because options are saved in your "
            f"shell history."
        )


def run(
    args: argparse.Namespace,
    *,
    out: TextIO,
    err: TextIO,
    stdin: Optional[TextIO] = None,
) -> int:
    stdin = sys.stdin if stdin is None else stdin
    _require_api_key()

    if args.command == "deploy":
        if not args.agent:
            raise UsageError(f"No agent given. Pass --agent, or set {ENV_AGENT_ID}.")
        upload = read_upload(args.path, out)
        with Client(base_url=args.base_url) as client:
            return deploy(
                client,
                path=args.path,
                upload=upload,
                agent_id=args.agent,
                out=out,
                err=err,
                sleep=time.sleep,
                monotonic=time.monotonic,
                interactive=out.isatty(),
                base_url=args.base_url,
            )

    if args.command == "secrets":
        if args.subcommand == "set":
            check_secret_name(args.name)
            value = read_secret_value(args.name, stdin, stdin.isatty())
            with Client(base_url=args.base_url) as client:
                return secrets_set(
                    client, name=args.name, value=value, out=out, err=err
                )
        if args.subcommand == "list":
            with Client(base_url=args.base_url) as client:
                return secrets_list(
                    client, out=out, err=err, interactive=out.isatty()
                )
        if args.subcommand == "delete":
            check_secret_name(args.name)
            with Client(base_url=args.base_url) as client:
                return secrets_delete(client, name=args.name, out=out, err=err)

    if args.command == "deployments":
        if args.subcommand == "list":
            with Client(base_url=args.base_url) as client:
                return deployments_list(
                    client,
                    agent_id=args.agent,
                    limit=args.limit,
                    fetch_all=args.fetch_all,
                    out=out,
                    err=err,
                    interactive=out.isatty(),
                )
        if args.subcommand == "status":
            with Client(base_url=args.base_url) as client:
                return deployments_status(
                    client, deployment_id=args.deployment_id, out=out, err=err
                )
        if args.subcommand == "delete":
            with Client(base_url=args.base_url) as client:
                return deployments_delete(
                    client, deployment_id=args.deployment_id, out=out, err=err
                )

    raise UsageError(f"Unknown command: {args.command}")


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args, out=sys.stdout, err=sys.stderr)
    except UsageError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return INTERRUPTED_EXIT_CODE


if __name__ == "__main__":
    sys.exit(main())
