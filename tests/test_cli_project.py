import base64
import io
import json
import os
import tarfile

import pytest
from assemblyai_agents import _cli, _project

AGENT_ID = "agent_b4c9e0d27a314c6e9f5a8d2e6c1b0a47"
DEPLOYMENT_ID = "agentdep_cc3b6476a05949878814407b9b4da456"

ENTRY = (
    "from assemblyai_agents import tool\n"
    "\n"
    "from pkg.deep.arithmetic import checksum\n"
    "\n"
    "\n"
    "@tool()\n"
    "def order_checksum(order_id: str) -> dict:\n"
    '    """Doc."""\n'
    "    return {'checksum': checksum(order_id)}\n"
)
NESTED = "def checksum(value: str) -> str:\n    return value[::-1]\n"


def write(root, relative: str, body: str = "") -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def make_project(root):
    """The shape the live proof deploys: an entry point over a nested package.

    Written in an order that is not the order it packs in, so a walk that
    happened to preserve creation order would not pass the sorting assertions.
    """
    write(root, "pkg/deep/arithmetic.py", NESTED)
    write(root, "main.py", ENTRY)
    write(root, "pkg/deep/__init__.py")
    write(root, "pkg/__init__.py")
    return root


def names(archive: bytes):
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        return [member.name for member in tar]


def packed(root) -> bytes:
    project = _project.read_project(root)
    return _project.build_archive(project.files)


# ------------------------------------------------------------ what travels


def test_a_nested_project_packs_every_file_under_its_own_path(tmp_path):
    archive = packed(make_project(tmp_path))

    assert names(archive) == [
        "main.py",
        "pkg/__init__.py",
        "pkg/deep/__init__.py",
        "pkg/deep/arithmetic.py",
    ]


def test_the_entry_point_arrives_with_its_contents_intact(tmp_path):
    archive = packed(make_project(tmp_path))

    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        body = tar.extractfile("pkg/deep/arithmetic.py").read()

    assert body.decode("utf-8") == NESTED


def test_dependency_files_are_never_excluded(tmp_path):
    """The thing a wrong ignore rule breaks silently: the project builds without
    the packages it declares, and nothing says why."""
    make_project(tmp_path)
    write(tmp_path, "pyproject.toml", "[project]\nname = 'tools'\n")
    write(tmp_path, "requirements.txt", "httpx==0.27.0\n")
    write(tmp_path, "uv.lock", "version = 1\n")

    assert {"pyproject.toml", "requirements.txt", "uv.lock"} <= set(
        names(packed(tmp_path))
    )


# --------------------------------------------------------------- exclusions


def test_env_files_are_left_behind_and_named(tmp_path):
    make_project(tmp_path)
    write(tmp_path, ".env", "ORDERS_API_KEY=sk-live-do-not-ship\n")
    write(tmp_path, ".env.production", "ORDERS_API_KEY=sk-live-either\n")

    project = _project.read_project(tmp_path)
    archive = _project.build_archive(project.files)

    assert names(archive) == [
        "main.py",
        "pkg/__init__.py",
        "pkg/deep/__init__.py",
        "pkg/deep/arithmetic.py",
    ]
    assert b"sk-live-do-not-ship" not in archive
    assert {".env", ".env.production"} == {
        name for name, _ in project.excluded if name.startswith(".env")
    }


def test_the_control_shows_a_packed_secret_would_be_visible(tmp_path):
    """The red half of the test above: the same value in a file that is not a
    .env does travel, so the assertion is reading the archive, not a typo."""
    make_project(tmp_path)
    write(tmp_path, "config.py", "ORDERS_API_KEY = 'sk-live-do-not-ship'\n")

    assert b"sk-live-do-not-ship" in packed(tmp_path)


def test_build_output_and_repository_metadata_do_not_travel(tmp_path):
    make_project(tmp_path)
    write(tmp_path, "__pycache__/main.cpython-311.pyc", "x")
    write(tmp_path, "pkg/stale.pyc", "x")
    write(tmp_path, ".git/config", "[core]\n")
    write(tmp_path, ".mypy_cache/index.json", "{}")

    assert names(packed(tmp_path)) == [
        "main.py",
        "pkg/__init__.py",
        "pkg/deep/__init__.py",
        "pkg/deep/arithmetic.py",
    ]


def test_a_virtual_environment_is_found_by_its_marker_not_its_name(tmp_path):
    make_project(tmp_path)
    write(tmp_path, "whatever/pyvenv.cfg", "home = /usr\n")
    write(tmp_path, "whatever/lib/site-packages/httpx/__init__.py", "x = 1\n")

    project = _project.read_project(tmp_path)

    assert "whatever/lib/site-packages/httpx/__init__.py" not in project.files
    assert ("whatever/", "a virtual environment") in project.excluded


def test_a_package_genuinely_called_venv_still_travels(tmp_path):
    """The control for the rule above: matching on the name would drop this."""
    make_project(tmp_path)
    write(tmp_path, "venv/__init__.py", "NAME = 'a real package'\n")

    assert "venv/__init__.py" in _project.read_project(tmp_path).files


def test_the_ignore_file_excludes_what_it_names_and_itself(tmp_path):
    make_project(tmp_path)
    write(tmp_path, _project.IGNORE_FILE, "# notes\nfixtures\n*.csv\n")
    write(tmp_path, "fixtures/orders.json", "[]")
    write(tmp_path, "pkg/sample.csv", "a,b\n")

    packed_names = names(packed(tmp_path))

    assert "fixtures/orders.json" not in packed_names
    assert "pkg/sample.csv" not in packed_names
    assert _project.IGNORE_FILE not in packed_names


def test_without_the_ignore_file_those_same_files_travel(tmp_path):
    make_project(tmp_path)
    write(tmp_path, "fixtures/orders.json", "[]")
    write(tmp_path, "pkg/sample.csv", "a,b\n")

    packed_names = names(packed(tmp_path))

    assert "fixtures/orders.json" in packed_names
    assert "pkg/sample.csv" in packed_names


def test_a_directory_with_no_files_is_reported(tmp_path):
    make_project(tmp_path)
    (tmp_path / "pkg" / "hollow").mkdir()

    assert _project.read_project(tmp_path).empty_directories == ["pkg/hollow"]


# ------------------------------------------------------------- determinism


def test_the_same_tree_packs_to_the_same_bytes_twice(tmp_path):
    root = make_project(tmp_path)

    first = packed(root)
    second = packed(root)

    assert first == second
    # Two empty trees would also agree. This is what says the bytes compared
    # are a real project's.
    assert len(names(first)) == 4


def test_two_copies_differing_only_in_timestamps_pack_identically(tmp_path):
    left = make_project(tmp_path / "left")
    right = make_project(tmp_path / "right")
    for path in right.rglob("*"):
        os.utime(path, (1_700_000_000, 1_700_000_000))
    os.utime(right, (1_700_000_000, 1_700_000_000))

    assert packed(left) == packed(right)


def test_a_changed_file_changes_the_bytes(tmp_path):
    """The red control for the two tests above."""
    root = make_project(tmp_path)
    before = packed(root)
    write(root, "pkg/deep/arithmetic.py", NESTED + "\n# one more line\n")

    assert packed(root) != before


def test_the_tar_is_sorted_even_when_the_walk_is_not(tmp_path):
    """A top-level file that sorts after a subdirectory's contents.

    Without one the walk hands back names that are already in order, so an
    assertion that they are sorted holds whether the packing sorts them or not.
    """
    make_project(tmp_path)
    write(tmp_path, "zz_last.py", "LAST = 1\n")

    assert names(packed(tmp_path)) == [
        "main.py",
        "pkg/__init__.py",
        "pkg/deep/__init__.py",
        "pkg/deep/arithmetic.py",
        "zz_last.py",
    ]


def test_every_field_a_tar_records_about_time_or_owner_is_fixed(tmp_path):
    archive = packed(make_project(tmp_path))

    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        members = list(tar)

    assert [member.name for member in members] == sorted(
        member.name for member in members
    )
    for member in members:
        assert (member.mtime, member.uid, member.gid) == (0, 0, 0)
        assert (member.uname, member.gname) == ("", "")
        assert member.mode == 0o644
        assert member.isfile()


# ---------------------------------------------------------------- refusals


def test_a_symlink_pointing_out_of_the_project_is_refused(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "id_rsa").write_text("PRIVATE KEY", encoding="utf-8")
    root = make_project(tmp_path / "project")
    (root / "borrowed.py").symlink_to(outside / "id_rsa")

    with pytest.raises(_project.ProjectError) as caught:
        _project.read_project(root)

    assert "outside" in str(caught.value)
    assert "id_rsa" in str(caught.value)


def test_a_symlink_to_a_directory_outside_the_project_is_refused(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secrets.py").write_text("KEY = 'x'", encoding="utf-8")
    root = make_project(tmp_path / "project")
    (root / "linked").symlink_to(outside, target_is_directory=True)

    with pytest.raises(_project.ProjectError) as caught:
        _project.read_project(root)

    assert "outside" in str(caught.value)


def test_a_symlink_inside_the_project_is_refused_too(tmp_path):
    root = make_project(tmp_path)
    (root / "alias.py").symlink_to(root / "pkg" / "deep" / "arithmetic.py")

    with pytest.raises(_project.ProjectError) as caught:
        _project.read_project(root)

    assert "symbolic link" in str(caught.value)


def test_the_same_project_without_the_link_packs_cleanly(tmp_path):
    """The control for the three refusals above."""
    assert len(names(packed(make_project(tmp_path)))) == 4


def test_a_file_that_is_not_text_is_refused_by_name(tmp_path):
    root = make_project(tmp_path)
    (root / "pkg" / "fast.so").write_bytes(b"\x7fELF\x02\x01\x01\x00\xff\xfe")

    with pytest.raises(_project.ProjectError) as caught:
        _project.read_project(root)

    assert "pkg/fast.so" in str(caught.value)
    assert "UTF-8" in str(caught.value)


def test_a_project_without_an_entry_point_is_refused(tmp_path):
    write(tmp_path, "tools.py", ENTRY)

    with pytest.raises(_project.ProjectError) as caught:
        _project.read_project(tmp_path)

    assert _project.ENTRY_NAME in str(caught.value)


def test_an_empty_directory_is_refused(tmp_path):
    with pytest.raises(_project.ProjectError) as caught:
        _project.read_project(tmp_path)

    assert "no files" in str(caught.value)


def test_more_files_than_the_service_stores_is_refused(tmp_path):
    make_project(tmp_path)
    for index in range(_project.MAX_FILES + 1):
        write(tmp_path, f"many/module_{index:04d}.py", "x = 1\n")

    with pytest.raises(_project.ProjectError) as caught:
        _project.read_project(tmp_path)

    assert str(_project.MAX_FILES) in str(caught.value)


def test_one_file_under_the_ceiling_is_accepted(tmp_path):
    """The control: the count is the limit, not the fixture being large."""
    make_project(tmp_path)
    for index in range(_project.MAX_FILES - 5):
        write(tmp_path, f"many/module_{index:04d}.py", "x = 1\n")

    assert len(_project.read_project(tmp_path).files) == _project.MAX_FILES - 1


def test_a_project_larger_than_the_upload_limit_is_refused(tmp_path):
    make_project(tmp_path)
    write(tmp_path, "bulk/catalogue.py", "# " + "x" * _project.MAX_ARCHIVE_BYTES)

    project = _project.read_project(tmp_path)
    with pytest.raises(_project.ProjectError) as caught:
        _project.build_archive(project.files)

    assert "bulk/catalogue.py" in str(caught.value)
    assert f"{_project.MAX_ARCHIVE_BYTES:,}" in str(caught.value)


def test_a_project_just_under_the_upload_limit_is_accepted(tmp_path):
    """The control for the refusal above."""
    make_project(tmp_path)
    write(tmp_path, "bulk/catalogue.py", "# " + "x" * (3 * 1024 * 1024))

    archive = packed(tmp_path)

    assert _project.MAX_ARCHIVE_BYTES > len(archive) > 3 * 1024 * 1024


def test_a_path_too_long_to_store_is_refused(tmp_path):
    make_project(tmp_path)
    write(tmp_path, "deep/" + "a" * _project.MAX_PATH_CHARACTERS + ".py", "x = 1\n")

    with pytest.raises(_project.ProjectError) as caught:
        _project.read_project(tmp_path)

    assert str(_project.MAX_PATH_CHARACTERS) in str(caught.value)


def test_a_pipe_in_the_tree_is_refused(tmp_path):
    root = make_project(tmp_path)
    os.mkfifo(root / "feed")

    with pytest.raises(_project.ProjectError) as caught:
        _project.read_project(root)

    assert "regular file" in str(caught.value)


# ---------------------------------------------------- what the CLI sends


def _responses():
    created = (
        201,
        {
            "id": DEPLOYMENT_ID,
            "agent_id": AGENT_ID,
            "status": "pending",
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:00:00Z",
        },
        None,
    )
    ready = (
        200,
        {
            "id": DEPLOYMENT_ID,
            "agent_id": AGENT_ID,
            "status": "ready",
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:00:00Z",
        },
        None,
    )
    return [created, ready]


def _deploy(make_client, recorder, path, out):
    client = make_client(_responses(), recorder)
    return _cli.deploy(
        client,
        path=str(path),
        upload=_cli.read_upload(str(path), out),
        agent_id=AGENT_ID,
        out=out,
        err=io.StringIO(),
        sleep=lambda _seconds: None,
        monotonic=lambda: 0.0,
        interactive=False,
    )


def test_deploying_a_directory_puts_the_tar_in_the_archive_field(
    tmp_path, make_client, recorder
):
    make_project(tmp_path)
    out = io.StringIO()

    code = _deploy(make_client, recorder, tmp_path, out)

    body = json.loads(recorder.requests[0].content)
    assert code == 0
    assert set(body) == {"agent_id", "deployment_type", "archive"}
    assert names(base64.b64decode(body["archive"])) == [
        "main.py",
        "pkg/__init__.py",
        "pkg/deep/__init__.py",
        "pkg/deep/arithmetic.py",
    ]
    assert "Packed 4 files" in out.getvalue()


def test_deploying_one_file_still_sends_source(tmp_path, make_client, recorder):
    module = tmp_path / "tools.py"
    module.write_text(ENTRY, encoding="utf-8")
    out = io.StringIO()

    code = _deploy(make_client, recorder, module, out)

    body = json.loads(recorder.requests[0].content)
    assert code == 0
    assert body == {
        "agent_id": AGENT_ID,
        "deployment_type": "tools",
        "source": ENTRY,
    }


def test_a_project_that_cannot_be_packed_never_reaches_the_api(
    tmp_path, make_client, recorder
):
    root = make_project(tmp_path)
    (root / "alias.py").symlink_to(root / "main.py")

    with pytest.raises(_cli.UsageError):
        _cli.read_upload(str(root), io.StringIO())

    assert recorder.count == 0


def test_the_excluded_env_file_is_printed_with_where_secrets_belong(tmp_path):
    make_project(tmp_path)
    write(tmp_path, ".env", "KEY=value\n")
    out = io.StringIO()

    _cli.read_upload(str(tmp_path), out)

    assert ".env" in out.getvalue()
    assert "secrets set" in out.getvalue()


def test_a_nested_env_file_still_gets_the_pointer_to_secrets(tmp_path):
    """The reason, not the path, is what the pointer keys off: a `.env` two
    directories down is the same mistake as one at the top."""
    make_project(tmp_path)
    write(tmp_path, "pkg/deep/.env", "KEY=value\n")
    out = io.StringIO()

    _cli.read_upload(str(tmp_path), out)

    assert "pkg/deep/.env" in out.getvalue()
    assert "secrets set" in out.getvalue()
