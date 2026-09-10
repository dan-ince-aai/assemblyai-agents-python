import inspect

import pytest
from assemblyai_agents import _context
from assemblyai_agents._context import (
    HTTP_TIMEOUT_SECONDS,
    RESPONSE_LIMIT_BYTES,
    ToolContext,
)


class Double:
    def __init__(self) -> None:
        self.http = object()
        self.log = object()
        self.session_id = "sess_1"
        self.aborted = False

    def secret(self, name: str) -> str:
        return "test-key"


class MissingSecret:
    def __init__(self) -> None:
        self.http = object()
        self.log = object()
        self.session_id = "sess_1"
        self.aborted = False


def test_a_complete_double_satisfies_the_protocol():
    assert isinstance(Double(), ToolContext)


def test_a_double_missing_a_member_does_not_satisfy_the_protocol():
    assert not isinstance(MissingSecret(), ToolContext)


def test_the_protocol_declares_the_five_members():
    members = set(ToolContext.__annotations__) | {"secret"}

    assert members == {"http", "log", "session_id", "aborted", "secret"}
    assert inspect.signature(ToolContext.secret).parameters.keys() == {"self", "name"}


def test_the_protocol_has_no_implementation():
    # v1 ships the shape only: the concrete double lands in `testing.py` and the
    # egress client is the runtime's, not this package's.
    with pytest.raises(TypeError):
        ToolContext()

    declared_here = [
        name
        for name, value in vars(_context).items()
        if inspect.isclass(value) and value.__module__ == _context.__name__
    ]
    assert declared_here == ["ToolContext"]
    assert not hasattr(_context, "httpx")


def test_the_settled_limits():
    assert RESPONSE_LIMIT_BYTES == 1048576
    assert RESPONSE_LIMIT_BYTES == 1024 * 1024
    assert HTTP_TIMEOUT_SECONDS == 15.0


def test_the_protocol_is_runtime_checkable():
    # An isinstance check against a Protocol without @runtime_checkable raises, so
    # the two checks above already depend on it; this names the requirement.
    assert getattr(ToolContext, "_is_runtime_protocol", False)
