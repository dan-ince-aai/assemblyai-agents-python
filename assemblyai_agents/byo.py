"""Deprecated alias for :mod:`assemblyai_agents.replies`.

The module was renamed: `byo` said how the endpoint is wired ("bring your
own"), where `replies` says what the module is for. Import from
``assemblyai_agents.replies``; this alias will be removed in a later release.
"""

from .replies import (  # noqa: F401
    PRE_CONNECT_TOOL,
    Call,
    Say,
    Silence,
    ToolResult,
    Turn,
    call_tool,
    digits_said,
    established,
    json_body,
    say,
    silence,
    stream,
)
