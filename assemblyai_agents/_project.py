import fnmatch
import io
import os
import posixpath
import tarfile
from pathlib import Path
from typing import Dict, List, NamedTuple, Tuple

# The name the entry point has inside the upload. AssemblyAI reads a tools
# deployment's tools, and a service deployment's application, from this file,
# and imports it under a name of its own, so a sibling module can import the
# other siblings but cannot import this one.
ENTRY_NAME = "main.py"

# Every limit below is the one the API enforces, checked here so an oversized
# project is refused in milliseconds instead of after a multi-megabyte upload.
MAX_ARCHIVE_BYTES = 4 * 1024 * 1024
MAX_UNPACKED_BYTES = 32 * 1024 * 1024
MAX_FILES = 512
MAX_PATH_CHARACTERS = 200

IGNORE_FILE = ".assemblyaiignore"

# Dropped wherever they appear. The first four are what the API drops too, so
# what is counted against the size limits here is what it stores; the rest are
# tooling output that is never a module the customer wrote.
SKIPPED_DIRECTORIES = frozenset(
    {
        "__pycache__",
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".eggs",
        ".idea",
        ".vscode",
        "node_modules",
    }
)
SKIPPED_SUFFIXES = (".pyc", ".pyo")

# What marks a directory as a virtual environment, whatever it has been named.
# Matching on the name would either miss `env3/` or swallow a package genuinely
# called `venv`; this file is present in every environment and in nothing else.
VENV_MARKER = "pyvenv.cfg"

# Named, because the CLI adds a line about where credentials do belong whenever
# one of these was left behind, and matching on the path would miss a nested one.
ENV_FILE_REASON = "an environment file, which never travels"


class ProjectError(Exception):
    """The directory cannot be deployed. The message is shown to the customer."""


class Project(NamedTuple):
    files: Dict[str, bytes]
    # (path, why), in walk order, for printing. A pruned directory appears once
    # rather than once per file inside it.
    excluded: List[Tuple[str, str]]
    # Directories that contributed nothing. An upload is a set of file paths, so
    # a directory with no files in it cannot be represented at the far end.
    empty_directories: List[str]


def read_project(root: Path) -> Project:
    patterns = _ignore_patterns(root)
    files: Dict[str, bytes] = {}
    excluded: List[Tuple[str, str]] = []
    populated = set()
    visited = []
    unpacked = 0

    for directory, names, filenames in os.walk(root, topdown=True, followlinks=False):
        here = _relative(root, Path(directory))
        names.sort()
        filenames.sort()

        if here and VENV_MARKER in filenames:
            names[:] = []
            excluded.append((here + "/", "a virtual environment"))
            continue

        kept = []
        for name in names:
            child = posixpath.join(here, name) if here else name
            path = Path(directory) / name
            if path.is_symlink():
                raise ProjectError(_symlink_message(root, path, child))
            if name in SKIPPED_DIRECTORIES:
                excluded.append((child + "/", f"`{name}` never travels"))
                continue
            if _ignored(child, patterns):
                excluded.append((child + "/", f"matched by {IGNORE_FILE}"))
                continue
            kept.append(name)
            visited.append(child)
        names[:] = kept

        for name in filenames:
            child = posixpath.join(here, name) if here else name
            path = Path(directory) / name
            if path.is_symlink():
                raise ProjectError(_symlink_message(root, path, child))
            reason = _skip_reason(here, name, child, patterns)
            if reason is not None:
                excluded.append((child, reason))
                continue
            if not path.is_file():
                raise ProjectError(
                    f"`{child}` is not a regular file. A deployment is a tree of "
                    f"source files, so a device, socket or pipe cannot be part of "
                    f"one."
                )
            _check_path(child)
            body = _read_text_file(path, child)
            unpacked += len(body)
            if unpacked > MAX_UNPACKED_BYTES:
                raise ProjectError(
                    f"The project holds more than "
                    f"{MAX_UNPACKED_BYTES // (1024 * 1024)} MiB of files, which "
                    f"is the most AssemblyAI stores."
                )
            if len(files) >= MAX_FILES:
                raise ProjectError(
                    f"The project holds more than {MAX_FILES} files, which is "
                    f"the most AssemblyAI stores. Exclude what the deployment does not "
                    f"need with a {IGNORE_FILE} file beside {ENTRY_NAME}."
                )
            files[child] = body
            parent = posixpath.dirname(child)
            while parent:
                populated.add(parent)
                parent = posixpath.dirname(parent)

    if not files:
        raise ProjectError(
            f"{root} holds no files to deploy. It needs a {ENTRY_NAME} holding "
            f"your tools, or your application."
        )
    if ENTRY_NAME not in files:
        raise ProjectError(
            f"{root} has no {ENTRY_NAME}. That file is the one your tools — or, "
            f"for a service, your application — are read from, and it has to "
            f"sit at the top of the project directory."
        )
    return Project(
        files=files,
        excluded=excluded,
        empty_directories=[name for name in visited if name not in populated],
    )


def build_archive(files: Dict[str, bytes]) -> bytes:
    """One uncompressed tar, byte-for-byte reproducible from the same tree.

    Uncompressed, and with every field that records a time or an owner fixed,
    because AssemblyAI names the stored project and its image after a hash of
    exactly these bytes. Anything left to vary — the order a directory happens
    to be walked in, the mtime of a file that was only touched, the compression
    library on the machine doing the deploying — would rebuild an image that is
    already built and hand back a new name for an unchanged project.
    """
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for name in sorted(files):
            body = files[name]
            info = tarfile.TarInfo(name)
            info.size = len(body)
            info.mode = 0o644
            info.mtime = 0
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            tar.addfile(info, io.BytesIO(body))
    archive = buffer.getvalue()
    if len(archive) > MAX_ARCHIVE_BYTES:
        raise ProjectError(
            f"The project packs to {len(archive):,} bytes, over the "
            f"{MAX_ARCHIVE_BYTES:,} AssemblyAI accepts. The largest files in it "
            f"are:\n{_largest(files)}"
        )
    return archive


def _relative(root: Path, path: Path) -> str:
    relative = path.relative_to(root).as_posix()
    return "" if relative == "." else relative


def _skip_reason(here: str, name: str, child: str, patterns: List[str]):
    # Checked before anything is read, so a credential file is never opened, let
    # alone packed.
    if name == ".env" or name.startswith(".env."):
        return ENV_FILE_REASON
    if name.endswith(SKIPPED_SUFFIXES):
        return "compiled Python"
    if not here and name == IGNORE_FILE:
        return "the ignore file itself"
    if _ignored(child, patterns):
        return f"matched by {IGNORE_FILE}"
    return None


def _ignore_patterns(root: Path) -> List[str]:
    path = root / IGNORE_FILE
    if not path.is_file():
        return []
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ProjectError(f"Cannot read {path}: {exc}.")
    patterns = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            patterns.append(stripped.rstrip("/"))
    return patterns


def _ignored(relative: str, patterns: List[str]) -> bool:
    """Glob matching, deliberately simpler than git's.

    A pattern with no slash matches any path segment by that name; a pattern
    with one matches the path from the top of the project, and covers whatever
    is underneath it. `fnmatchcase` rather than `fnmatch`, so a project does not
    pack differently on a case-insensitive filesystem.
    """
    for pattern in patterns:
        if "/" in pattern:
            if fnmatch.fnmatchcase(relative, pattern):
                return True
            if relative.startswith(pattern + "/"):
                return True
        elif any(fnmatch.fnmatchcase(part, pattern) for part in relative.split("/")):
            return True
    return False


def _check_path(relative: str) -> None:
    if len(relative) > MAX_PATH_CHARACTERS:
        raise ProjectError(
            f"`{relative}` is longer than the {MAX_PATH_CHARACTERS} characters "
            f"AssemblyAI stores a path in."
        )
    if "\\" in relative:
        raise ProjectError(
            f"`{relative}` has a backslash in its name, which AssemblyAI cannot "
            f"store as a path."
        )


def _read_text_file(path: Path, relative: str) -> bytes:
    try:
        body = path.read_bytes()
    except OSError as exc:
        raise ProjectError(f"Cannot read {relative}: {exc.strerror}.")
    try:
        body.decode("utf-8")
    except UnicodeDecodeError:
        raise ProjectError(
            f"`{relative}` is not UTF-8 text. AssemblyAI reads a deployment as "
            f"source files, so a compiled extension, an archive or an image "
            f"cannot be part of one. Remove it, or exclude it with a "
            f"{IGNORE_FILE} file beside {ENTRY_NAME}."
        )
    return body


def _symlink_message(root: Path, path: Path, relative: str) -> str:
    # `os.path.realpath` and not `Path.resolve(strict=True)`: a link to a
    # missing target still has to be reported by where it points.
    target = os.path.realpath(path)
    try:
        Path(target).relative_to(os.path.realpath(root))
    except ValueError:
        return (
            f"`{relative}` is a symbolic link to {target}, which is outside "
            f"{root}. A deployment may only hold files from the project "
            f"directory itself."
        )
    return (
        f"`{relative}` is a symbolic link. AssemblyAI stores a deployment as "
        f"plain files, so a link would arrive meaning something different from "
        f"what it means here. Copy what it points at into its place."
    )


def _largest(files: Dict[str, bytes]) -> str:
    ranked = sorted(files.items(), key=lambda item: -len(item[1]))[:5]
    return "\n".join(f"  {len(body):>12,}  {name}" for name, body in ranked)
