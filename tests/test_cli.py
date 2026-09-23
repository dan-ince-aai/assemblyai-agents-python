import io
import json

import pytest
from assemblyai_agents import _cli

from .conftest import Recorder, err

AGENT_ID = "agent_b4c9e0d27a314c6e9f5a8d2e6c1b0a47"
DEPLOYMENT_ID = "agentdep_cc3b6476a05949878814407b9b4da456"
SOURCE = 'from assemblyai_agents import tool\n\n\n@tool()\ndef f() -> dict:\n    """Doc."""\n    return {}\n'

TRACEBACK = (
    "Traceback (most recent call last):\n"
    '  File "tools.py", line 3, in <module>\n'
    "    import pytz\n"
    "ModuleNotFoundError: No module named 'pytz'"
)


class FakeClock:
    """Wall-clock stand-in whose only movement is the CLI's own poll sleeps."""

    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class FakeStdin(io.StringIO):
    """Standard input whose terminal-ness the test chooses."""

    def __init__(self, text: str = "", tty: bool = False) -> None:
        super().__init__(text)
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


class FakeStdout(io.StringIO):
    def __init__(self, tty: bool = False) -> None:
        super().__init__()
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


def _deployment(status: str, detail=None) -> dict:
    return {
        "id": DEPLOYMENT_ID,
        "agent_id": AGENT_ID,
        "status": status,
        "detail": detail,
        "created_at": "2026-09-18T10:00:00Z",
        "updated_at": "2026-09-18T10:00:00Z",
    }


def _created(status: str = "pending"):
    return (201, _deployment(status), None)


def _polled(status: str, detail=None):
    return (200, _deployment(status, detail), None)


def _run_deploy(make_client, recorder: Recorder, responses: list, **overrides):
    out = io.StringIO()
    errs = io.StringIO()
    clock = FakeClock()
    client = make_client(responses, recorder)
    code = _cli.deploy(
        client,
        path=overrides.get("path", "tools.py"),
        source=overrides.get("source", SOURCE),
        agent_id=AGENT_ID,
        out=out,
        err=errs,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
        interactive=overrides.get("interactive", False),
        base_url=overrides.get("base_url", "https://agents.test.local"),
    )
    return code, out.getvalue(), errs.getvalue(), clock


# ---------------------------------------------------------------- deploy


def test_successful_deploy_uploads_the_source_and_exits_zero(make_client, recorder):
    code, out, errs, _ = _run_deploy(
        make_client, recorder, [_created(), _polled("ready")]
    )

    assert code == 0
    post = recorder.requests[0]
    assert post.method == "POST"
    assert post.url.path == "/v1/agent-deployments"
    assert json.loads(post.content) == {"agent_id": AGENT_ID, "source": SOURCE}
    assert "Deployed in" in out
    assert AGENT_ID in out
    assert "tools.py" in out
    assert errs == ""


def test_the_success_line_says_assemblyai_runs_the_code(make_client, recorder):
    _, out, _, _ = _run_deploy(make_client, recorder, [_created(), _polled("ready")])

    assert "AssemblyAI is now running the tools in tools.py" in out


def test_polls_the_single_deployment_route_until_it_settles(make_client, recorder):
    code, out, _, _ = _run_deploy(
        make_client,
        recorder,
        [_created(), _polled("pending"), _polled("pending"), _polled("ready")],
    )

    assert code == 0
    polls = recorder.requests[1:]
    assert len(polls) == 3
    assert all(r.method == "GET" for r in polls)
    assert all(r.url.path == f"/v1/agent-deployments/{DEPLOYMENT_ID}" for r in polls)
    assert DEPLOYMENT_ID in out


def test_import_failure_prints_the_customers_own_error_and_exits_one(
    make_client, recorder
):
    code, out, errs, _ = _run_deploy(
        make_client,
        recorder,
        [_created(), _polled("import_failed", TRACEBACK)],
    )

    assert code == 1
    assert "import_failed" in errs
    assert "Your tool module did not import." in errs
    # The customer debugs against this text, so it has to arrive unaltered.
    assert TRACEBACK in errs
    assert DEPLOYMENT_ID in errs
    assert "Deployed in" not in out


def test_a_failed_dependency_install_is_explained(make_client, recorder):
    code, _, errs, _ = _run_deploy(
        make_client,
        recorder,
        [_created(), _polled("dependencies_failed", "No matching distribution")],
    )

    assert code == 1
    assert "dependencies could not be installed" in errs
    assert "No matching distribution" in errs


def test_a_status_this_version_does_not_know_keeps_polling(make_client, recorder):
    code, out, errs, _ = _run_deploy(
        make_client,
        recorder,
        [_created(), _polled("building"), _polled("building"), _polled("ready")],
    )

    assert code == 0
    assert "building" in out
    assert errs == ""


def test_giving_up_waiting_reports_the_deployment_is_still_running(
    make_client, recorder
):
    code, out, errs, clock = _run_deploy(
        make_client, recorder, [_created(), _polled("pending")]
    )

    assert code == 1
    assert _cli.WAIT_LIMIT_SECONDS <= clock.now < _cli.WAIT_LIMIT_SECONDS * 1.5
    assert "Gave up waiting" in errs
    assert "still pending" in errs
    assert DEPLOYMENT_ID in errs
    # The id is useless without the command that resolves it.
    assert f"deployments status {DEPLOYMENT_ID}" in errs


def test_a_long_wait_reports_progress_when_output_is_not_a_terminal(
    make_client, recorder
):
    responses = [_created()] + [_polled("pending")] * 12 + [_polled("ready")]

    code, out, _, clock = _run_deploy(make_client, recorder, responses)

    assert code == 0
    assert clock.now > _cli.HEARTBEAT_SECONDS
    assert "still pending" in out


def test_create_carries_an_idempotency_key(make_client, recorder):
    _run_deploy(make_client, recorder, [_created(), _polled("ready")])

    assert recorder.requests[0].headers.get("Idempotency-Key")


def test_a_rejected_create_is_reported_and_exits_one(make_client, recorder):
    code, out, errs, _ = _run_deploy(
        make_client,
        recorder,
        [(404, err("agent_not_found", request_id="req-42"), None)],
    )

    assert code == 1
    assert "agent_not_found" in errs
    assert "req-42" in errs
    assert "Created deployment" not in out


def test_an_unknown_agent_names_the_host_it_asked(make_client, recorder):
    _, _, errs, _ = _run_deploy(
        make_client,
        recorder,
        [(404, err("agent_not_found"), None)],
        base_url="https://agents.us.assemblyai.com",
    )

    # Agents are stored per host, so the wrong host is the commonest cause and
    # the message is useless without naming the one that was used.
    assert "https://agents.us.assemblyai.com" in errs


def test_a_capacity_refusal_says_it_is_a_concurrency_limit(make_client, recorder):
    _, _, errs, _ = _run_deploy(
        make_client,
        recorder,
        [(429, err("deployment_capacity_exceeded"), None)],
    )

    assert "running at once" in errs
    assert "deployments list" in errs


def test_an_error_while_polling_says_the_deployment_was_not_cancelled(
    make_client, recorder
):
    code, _, errs, _ = _run_deploy(
        make_client,
        recorder,
        [_created(), (404, err("agent_deployment_not_found"), None)],
    )

    assert code == 1
    assert "was not cancelled" in errs
    assert DEPLOYMENT_ID in errs


def test_interrupting_a_wait_says_the_deployment_is_still_running(
    make_client, recorder
):
    def interrupt(_seconds):
        raise KeyboardInterrupt

    out = io.StringIO()
    errs = io.StringIO()
    client = make_client([_created(), _polled("pending")], recorder)

    code = _cli.deploy(
        client,
        path="tools.py",
        source=SOURCE,
        agent_id=AGENT_ID,
        out=out,
        err=errs,
        sleep=interrupt,
        monotonic=FakeClock().monotonic,
        interactive=False,
    )

    assert code == _cli.INTERRUPTED_EXIT_CODE
    assert "Stopped waiting" in errs.getvalue()
    assert DEPLOYMENT_ID in errs.getvalue()


def test_source_larger_than_the_endpoint_accepts_is_refused_locally(tmp_path):
    module = tmp_path / "tools.py"
    module.write_text("# " + "x" * _cli.MAX_SOURCE_CHARACTERS, encoding="utf-8")

    with pytest.raises(_cli.UsageError) as exc_info:
        _cli.read_source(str(module))

    assert "at most" in str(exc_info.value)


def test_an_empty_module_is_refused_locally(tmp_path):
    module = tmp_path / "tools.py"
    module.write_text("\n  \n", encoding="utf-8")

    with pytest.raises(_cli.UsageError):
        _cli.read_source(str(module))


# ------------------------------------------------------------ secrets set


def test_setting_a_secret_puts_the_value_and_never_logs_it(make_client, recorder):
    out, errs = io.StringIO(), io.StringIO()
    client = make_client(
        [(200, {"name": "orders_key", "created_at": "c", "updated_at": "u"}, None)],
        recorder,
    )

    code = _cli.secrets_set(
        client, name="orders_key", value="s3cr3t", out=out, err=errs
    )

    assert code == 0
    put = recorder.requests[0]
    assert put.method == "PUT"
    assert put.url.path == "/v1/tool-secrets/orders_key"
    assert json.loads(put.content) == {"value": "s3cr3t"}
    assert "orders_key" in out.getvalue()
    # The whole point of stdin-only input is that the value never lands anywhere
    # a human or a log can read it back.
    assert "s3cr3t" not in out.getvalue()
    assert "s3cr3t" not in errs.getvalue()


def test_a_secret_value_comes_from_a_pipe_with_one_newline_stripped():
    assert _cli.read_secret_value("n", FakeStdin("abc\n"), False) == "abc"
    assert _cli.read_secret_value("n", FakeStdin("abc"), False) == "abc"
    # Only one, so a value that genuinely ends blank survives.
    assert _cli.read_secret_value("n", FakeStdin("abc\n\n"), False) == "abc\n"


def test_a_terminal_is_prompted_and_the_value_is_not_echoed(monkeypatch):
    asked = {}

    def fake_getpass(prompt):
        asked["prompt"] = prompt
        return "typed-value"

    monkeypatch.setattr(_cli.getpass, "getpass", fake_getpass)

    value = _cli.read_secret_value("orders_key", FakeStdin("", tty=True), True)

    assert value == "typed-value"
    assert "orders_key" in asked["prompt"]


def test_an_empty_value_is_refused_before_any_request():
    with pytest.raises(_cli.UsageError) as exc_info:
        _cli.read_secret_value("n", FakeStdin(""), False)

    assert "standard input" in str(exc_info.value)


def test_a_value_over_the_api_limit_is_refused_locally():
    too_long = "x" * (_cli.MAX_SECRET_VALUE_LENGTH + 1)

    with pytest.raises(_cli.UsageError) as exc_info:
        _cli.read_secret_value("n", FakeStdin(too_long), False)

    assert "at most" in str(exc_info.value)
    # The refusal must not quote the credential back.
    assert too_long not in str(exc_info.value)


@pytest.mark.parametrize(
    "name", ["orders-key", "9lives", "has space", "a.b", "", "x" * 65]
)
def test_a_name_the_api_would_reject_is_refused_locally(name):
    with pytest.raises(_cli.UsageError):
        _cli.check_secret_name(name)


@pytest.mark.parametrize("name", ["orders_key", "_private", "A1", "x" * 64])
def test_a_usable_name_is_accepted(name):
    _cli.check_secret_name(name)


def test_the_hyphen_refusal_explains_the_rule_rather_than_the_url():
    with pytest.raises(_cli.UsageError) as exc_info:
        _cli.check_secret_name("orders-key")

    message = str(exc_info.value)
    # The API enforces this as a URL path pattern, so its own refusal talks
    # about a path component. Ours has to talk about the name.
    assert "hyphen" in message
    assert "ctx.secret" in message


def test_the_account_cap_refusal_points_at_the_listing(make_client, recorder):
    out, errs = io.StringIO(), io.StringIO()
    client = make_client(
        [(422, err("tool_secret_limit_exceeded"), None)],
        recorder,
    )

    code = _cli.secrets_set(client, name="n", value="v", out=out, err=errs)

    assert code == 1
    assert "secrets list" in errs.getvalue()


# ----------------------------------------------------------- secrets list


def _secrets_page(names: list) -> tuple:
    return (
        200,
        {
            "secrets": [
                {"name": n, "created_at": "2026-09-01", "updated_at": "2026-09-02"}
                for n in names
            ]
        },
        None,
    )


def test_listing_secrets_prints_three_columns(make_client, recorder):
    out, errs = FakeStdout(tty=True), io.StringIO()
    client = make_client([_secrets_page(["a_key", "b_key"])], recorder)

    code = _cli.secrets_list(client, out=out, err=errs, interactive=True)

    assert code == 0
    lines = out.getvalue().splitlines()
    assert lines[0].split() == ["NAME", "CREATED", "UPDATED"]
    assert lines[1].split() == ["a_key", "2026-09-01", "2026-09-02"]


def test_the_header_is_omitted_when_output_is_piped(make_client, recorder):
    out, errs = io.StringIO(), io.StringIO()
    client = make_client([_secrets_page(["a_key"])], recorder)

    _cli.secrets_list(client, out=out, err=errs, interactive=False)

    # A header would become a bogus first record for awk or cut.
    assert "NAME" not in out.getvalue()
    assert out.getvalue().split()[0] == "a_key"


def test_an_empty_account_says_so_and_exits_zero(make_client, recorder):
    out, errs = io.StringIO(), io.StringIO()
    client = make_client([(200, {"secrets": []}, None)], recorder)

    code = _cli.secrets_list(client, out=out, err=errs, interactive=False)

    assert code == 0
    assert "No secrets" in out.getvalue()


# --------------------------------------------------------- secrets delete


def test_deleting_a_secret_sends_delete_and_survives_an_empty_body(
    make_client, recorder
):
    out, errs = io.StringIO(), io.StringIO()
    client = make_client([(204, None, None)], recorder)

    code = _cli.secrets_delete(client, name="orders_key", out=out, err=errs)

    assert code == 0
    assert recorder.requests[0].method == "DELETE"
    assert recorder.requests[0].url.path == "/v1/tool-secrets/orders_key"
    assert "Deleted orders_key" in out.getvalue()


def test_deleting_a_secret_that_is_not_there_exits_one(make_client, recorder):
    out, errs = io.StringIO(), io.StringIO()
    client = make_client([(404, err("tool_secret_not_found"), None)], recorder)

    code = _cli.secrets_delete(client, name="gone", out=out, err=errs)

    assert code == 1
    assert "no secret named gone" in errs.getvalue()


# -------------------------------------------------------- deployments list


def _deployments_page(ids: list, has_more: bool = False, cursor: str = "") -> tuple:
    return (
        200,
        {
            "agent_deployments": [
                {
                    "id": i,
                    "agent_id": AGENT_ID,
                    "status": "ready",
                    "created_at": "2026-09-18T10:00:00Z",
                    "updated_at": "2026-09-18T10:00:00Z",
                }
                for i in ids
            ],
            "has_more": has_more,
            "response_metadata": {"next_cursor": cursor},
        },
        None,
    )


def _run_list(make_client, recorder, responses, **kwargs):
    out, errs = io.StringIO(), io.StringIO()
    client = make_client(responses, recorder)
    code = _cli.deployments_list(
        client,
        agent_id=kwargs.get("agent_id"),
        limit=kwargs.get("limit"),
        fetch_all=kwargs.get("fetch_all", False),
        out=out,
        err=errs,
        interactive=kwargs.get("interactive", False),
    )
    return code, out.getvalue(), errs.getvalue()


def test_listing_deployments_prints_a_row_each(make_client, recorder):
    code, out, _ = _run_list(make_client, recorder, [_deployments_page(["d1", "d2"])])

    assert code == 0
    assert "d1" in out and "d2" in out
    assert recorder.requests[0].url.path == "/v1/agent-deployments"


def test_a_limit_is_sent_as_a_query_parameter(make_client, recorder):
    _run_list(make_client, recorder, [_deployments_page(["d1"])], limit=5)

    assert recorder.requests[0].url.params.get("limit") == "5"


def test_an_agent_filter_is_sent_and_marks_the_serving_deployment(
    make_client, recorder
):
    agent = (
        200,
        {"id": AGENT_ID, "tools": [{"name": "t", "deployment_id": "d2"}]},
        None,
    )
    code, out, _ = _run_list(
        make_client,
        recorder,
        [_deployments_page(["d1", "d2"]), agent],
        agent_id=AGENT_ID,
    )

    assert code == 0
    assert recorder.requests[0].url.params.get("agent_id") == AGENT_ID
    assert recorder.requests[1].url.path == f"/v1/agents/{AGENT_ID}"
    rows = {line.split()[0]: line for line in out.splitlines() if line.strip()}
    assert "(serving)" in rows["d2"]
    assert "(serving)" not in rows["d1"]


def test_all_follows_the_cursor(make_client, recorder):
    code, out, _ = _run_list(
        make_client,
        recorder,
        [
            _deployments_page(["d1"], has_more=True, cursor="c1"),
            _deployments_page(["d2"]),
        ],
        fetch_all=True,
    )

    assert code == 0
    assert "d1" in out and "d2" in out
    assert recorder.requests[1].url.params.get("cursor") == "c1"


def test_a_truncated_listing_says_there_are_more(make_client, recorder):
    _, out, _ = _run_list(
        make_client, recorder, [_deployments_page(["d1"], has_more=True, cursor="c1")]
    )

    assert "--all" in out


def test_an_empty_listing_exits_zero(make_client, recorder):
    code, out, _ = _run_list(make_client, recorder, [_deployments_page([])])

    assert code == 0
    assert "No deployments" in out


# ------------------------------------------------------ deployments status


def _run_status(make_client, recorder, responses):
    out, errs = io.StringIO(), io.StringIO()
    client = make_client(responses, recorder)
    code = _cli.deployments_status(
        client, deployment_id=DEPLOYMENT_ID, out=out, err=errs
    )
    return code, out.getvalue(), errs.getvalue()


def test_status_reads_the_single_deployment_route(make_client, recorder):
    _run_status(make_client, recorder, [_polled("ready")])

    # The single read is what applies the age check a listing skips, so it has
    # to be this route and not the listing.
    assert recorder.requests[0].url.path == f"/v1/agent-deployments/{DEPLOYMENT_ID}"


def test_a_ready_deployment_exits_zero(make_client, recorder):
    code, out, _ = _run_status(make_client, recorder, [_polled("ready")])

    assert code == 0
    assert "ready" in out


@pytest.mark.parametrize(
    "status",
    ["import_failed", "no_tools_found", "dependencies_failed", "timed_out",
     "internal_error"],
)
def test_a_failed_deployment_exits_one(make_client, recorder, status):
    code, _, _ = _run_status(make_client, recorder, [_polled(status)])

    assert code == 1


@pytest.mark.parametrize("status", ["pending", "building"])
def test_a_running_deployment_exits_three(make_client, recorder, status):
    code, _, _ = _run_status(make_client, recorder, [_polled(status)])

    assert code == _cli.PENDING_EXIT_CODE


def test_a_status_this_version_never_heard_of_is_not_called_a_failure(
    make_client, recorder
):
    code, _, _ = _run_status(make_client, recorder, [_polled("provisioning")])

    assert code == _cli.PENDING_EXIT_CODE


def test_status_prints_the_detail_in_full(make_client, recorder):
    _, out, _ = _run_status(
        make_client, recorder, [_polled("import_failed", TRACEBACK)]
    )

    assert TRACEBACK in out


def test_a_missing_deployment_exits_one(make_client, recorder):
    code, _, errs = _run_status(
        make_client, recorder, [(404, err("agent_deployment_not_found"), None)]
    )

    assert code == 1
    assert "agent_deployment_not_found" in errs


# ------------------------------------------------------ deployments delete


def _run_delete(make_client, recorder, responses):
    out, errs = io.StringIO(), io.StringIO()
    client = make_client(responses, recorder)
    code = _cli.deployments_delete(
        client, deployment_id=DEPLOYMENT_ID, out=out, err=errs
    )
    return code, out.getvalue(), errs.getvalue()


def test_deleting_a_deployment_sends_delete(make_client, recorder):
    code, out, _ = _run_delete(make_client, recorder, [(204, None, None)])

    assert code == 0
    assert recorder.requests[0].method == "DELETE"
    assert recorder.requests[0].url.path == f"/v1/agent-deployments/{DEPLOYMENT_ID}"
    assert "Deleted deployment" in out


def test_a_refusal_because_it_is_still_building_is_printed_verbatim(
    make_client, recorder
):
    message = (
        f"Deployment {DEPLOYMENT_ID} is still in progress and is about to become "
        f"this agent's tools. Delete it once it has finished."
    )
    code, _, errs = _run_delete(
        make_client,
        recorder,
        [(409, err("deployment_in_use", message=message), None)],
    )

    assert code == 1
    # The server distinguishes the two cases; rewording either loses the detail
    # that tells the customer what to do.
    assert message in errs


def test_a_refusal_because_an_agent_still_uses_it_is_printed_verbatim(
    make_client, recorder
):
    message = (
        f"Deployment {DEPLOYMENT_ID} still provides the tools on agent "
        f"{AGENT_ID}. Deploy again to that agent, or delete it, then delete "
        f"this deployment."
    )
    code, _, errs = _run_delete(
        make_client,
        recorder,
        [(409, err("deployment_in_use", message=message), None)],
    )

    assert code == 1
    assert message in errs
    # The agent id is the actionable part and only the server knows it.
    assert AGENT_ID in errs


def test_deleting_a_deployment_that_is_not_there_exits_one(make_client, recorder):
    code, _, errs = _run_delete(
        make_client, recorder, [(404, err("agent_deployment_not_found"), None)]
    )

    assert code == 1
    assert DEPLOYMENT_ID in errs


# ------------------------------------------------------------ parser rules


def test_a_missing_api_key_exits_two_without_calling_the_api(monkeypatch, capsys):
    monkeypatch.delenv("ASSEMBLYAI_API_KEY", raising=False)
    monkeypatch.delenv("ASSEMBLYAI_AGENT_ID", raising=False)

    code = _cli.main(["deploy", "tools.py", "--agent", AGENT_ID])

    assert code == 2
    assert "ASSEMBLYAI_API_KEY" in capsys.readouterr().err


def test_a_missing_agent_exits_two(monkeypatch, capsys):
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "k-test")
    monkeypatch.delenv("ASSEMBLYAI_AGENT_ID", raising=False)

    code = _cli.main(["deploy", "tools.py"])

    assert code == 2
    assert "ASSEMBLYAI_AGENT_ID" in capsys.readouterr().err


def test_the_api_key_cannot_be_passed_as_an_option():
    with pytest.raises(SystemExit):
        _cli.build_parser().parse_args(
            ["deploy", "tools.py", "--agent", AGENT_ID, "--api-key", "k-secret"]
        )


def test_a_secret_value_cannot_be_passed_as_an_option():
    with pytest.raises(SystemExit):
        _cli.build_parser().parse_args(["secrets", "set", "n", "--value", "v"])


def test_a_secret_value_cannot_be_passed_as_a_positional():
    with pytest.raises(SystemExit):
        _cli.build_parser().parse_args(["secrets", "set", "n", "v"])


@pytest.mark.parametrize(
    "argv",
    [
        ["deploy", "tools.py", "--agent", AGENT_ID],
        ["secrets", "set", "n"],
        ["secrets", "list"],
        ["secrets", "delete", "n"],
        ["deployments", "list"],
        ["deployments", "status", DEPLOYMENT_ID],
        ["deployments", "delete", DEPLOYMENT_ID],
    ],
)
def test_every_command_takes_base_url_after_the_subcommand(argv):
    # A top-level-only option would force `assemblyai-agents --base-url URL
    # secrets set NAME`, which nobody guesses.
    args = _cli.build_parser().parse_args(argv + ["--base-url", "https://x.test"])

    assert args.base_url == "https://x.test"


def test_a_bare_group_name_is_a_usage_error():
    for argv in (["secrets"], ["deployments"]):
        with pytest.raises(SystemExit):
            _cli.build_parser().parse_args(argv)


def test_a_limit_outside_the_api_range_is_refused_by_the_parser():
    for bad in ["0", "201", "abc"]:
        with pytest.raises(SystemExit):
            _cli.build_parser().parse_args(["deployments", "list", "--limit", bad])
