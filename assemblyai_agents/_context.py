from typing import Any, Protocol, runtime_checkable

RESPONSE_LIMIT_BYTES = 1048576

# The per-request timeout is the lesser of this and what is left of the tool's own
# budget. A flat 15s under a tool configured at 5s can never fire, so the request
# would die with no named reason and the author could not tell a slow endpoint from
# a slow tool.
HTTP_TIMEOUT_SECONDS = 15.0


@runtime_checkable
class ToolContext(Protocol):
    # `http` and `log` are untyped because the object that satisfies them lives
    # outside this package: the runtime supplies the egress client and the logger,
    # and the limits above are the contract that client has to meet.
    http: Any
    log: Any
    session_id: str
    aborted: bool

    def secret(self, name: str) -> str: ...
