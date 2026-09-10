import asyncio
import functools
import inspect
import re
from dataclasses import dataclass
from types import UnionType
from typing import (
    Any,
    Callable,
    Optional,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from pydantic import BaseModel

from ._context import ToolContext
from ._exceptions import ConfigurationError
from ._schema import _render, derive_schema
from .models.rest import (
    DtmfCollectionProfile,
    ExecutionMode,
    PlaintextHttpToolConfig,
    PlaintextToolDefinition,
    ResponseInstructions,
)

# The server's default for a stored tool. Set it lower on
# any tool a caller waits through: a running tool holds a worker slot, an STT
# socket, a TTS socket, a room and a sandbox for its whole duration, and past a
# few seconds the caller is listening to silence.
DEFAULT_TIMEOUT_SECONDS = 120
MIN_TIMEOUT_SECONDS = 1
MAX_TIMEOUT_SECONDS = 300

# The names of the AssemblyAI platform tools, copied here because the SDK ships
# as a standalone package. The server selects a tool's backend by NAME before it
# looks at anything else, so a customer tool under one of these names would route
# to the platform implementation and its own body would never run.
PLATFORM_TOOL_NAMES = frozenset(
    {"aai_credit_card_luhn_check", "aai_pre_connect_context"}
)

_SNAKE_CASE = re.compile(r"^[a-z][a-z0-9]*(_[a-z0-9]+)*$")

_JSON_RETURNS = "dict, list, str, int, float, bool, None, or a pydantic BaseModel"

_SECTIONS = (
    "Args",
    "Arguments",
    "Parameters",
    "Returns",
    "Return",
    "Yields",
    "Raises",
    "Example",
    "Examples",
    "Note",
    "Notes",
    "Attributes",
)
_SECTION = re.compile(rf"^({'|'.join(_SECTIONS)})\s*:\s*$")
_ARGS_SECTION = re.compile(r"^(Args|Arguments|Parameters)\s*:\s*$")
# `name (type): text` — the type is accepted and discarded, because the schema
# comes from the annotation and a docstring type would only be a second opinion.
_ARG = re.compile(r"^(\w+)\s*(?:\([^)]*\))?\s*:\s*(.*)$")


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict
    timeout_seconds: int
    execution_mode: Optional[ExecutionMode]
    response_instructions: Optional[ResponseInstructions]
    http: Optional[PlaintextHttpToolConfig]
    dtmf_collected_arguments: Optional[list[DtmfCollectionProfile]]
    context_parameter: Optional[str]
    is_async: bool
    target: Callable

    async def invoke(self, *, context: Any = None, **arguments) -> Any:
        if self.context_parameter is not None and context is not None:
            arguments[self.context_parameter] = context
        if self.is_async:
            return await self.target(**arguments)
        # A sync handler would hold the event loop for its whole timeout, and the
        # same loop is carrying the call's audio.
        return await asyncio.to_thread(self.target, **arguments)


class Tool:
    """What ``@tool`` returns: the handler, plus the wire definition it produces.

    Calling a ``Tool`` calls the function it decorates, unchanged — a sync
    handler returns its result, an async one returns an awaitable — so the module
    stays ordinary Python and the function is still directly callable from a
    test. ``definition()`` is the supported way to reach the wire model, and
    ``spec`` the supported way to reach the parsed description, schema and
    handler metadata.
    """

    def __init__(self, spec: ToolSpec) -> None:
        functools.update_wrapper(self, spec.target)
        self._spec = spec

    @property
    def name(self) -> str:
        return self._spec.name

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    @property
    def target(self) -> Callable:
        return self._spec.target

    def definition(self) -> PlaintextToolDefinition:
        spec = self._spec
        return PlaintextToolDefinition(
            name=spec.name,
            description=spec.description,
            parameters=spec.parameters,
            timeout_seconds=spec.timeout_seconds,
            # Left as None when it was never declared, so the payload omits it.
            # The stored agent always holds an explicit mode (the server fills
            # in its default at write time), so an absent mode and an explicit
            # `interactive` are identical once stored, and a later change to
            # the server default cannot move agents this SDK wrote.
            execution_mode=spec.execution_mode,
            response_instructions=spec.response_instructions,
            http=spec.http,
            dtmf_collected_arguments=spec.dtmf_collected_arguments,
        )

    async def invoke(self, *, context: Any = None, **arguments) -> Any:
        return await self._spec.invoke(context=context, **arguments)

    def __call__(self, *args, **kwargs) -> Any:
        return self._spec.target(*args, **kwargs)

    def __repr__(self) -> str:
        return f"Tool({self._spec.name!r})"


def tool(
    func: Optional[Callable] = None,
    *,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    execution_mode: Optional[ExecutionMode] = None,
    response_instructions: Optional[ResponseInstructions] = None,
    http: Optional[PlaintextHttpToolConfig] = None,
    dtmf_collected_arguments: Optional[list[DtmfCollectionProfile]] = None,
) -> Any:
    def decorate(target: Callable) -> Tool:
        return _declare(
            target,
            timeout_seconds=timeout_seconds,
            execution_mode=execution_mode,
            response_instructions=response_instructions,
            http=http,
            dtmf_collected_arguments=dtmf_collected_arguments,
        )

    if func is None:
        return decorate
    if not callable(func):
        raise ConfigurationError(
            f"@tool takes no name: a tool is named after its function, so the name "
            f"the model calls cannot drift from the name in the file. Got "
            f"{func!r}. Write `@tool` above a function called {func!r}."
        )
    return decorate(func)


def _declare(
    target: Callable,
    *,
    timeout_seconds: int,
    execution_mode: Any,
    response_instructions: Any,
    http: Any,
    dtmf_collected_arguments: Any,
) -> Tool:
    name = target.__name__
    _reject_platform_name(name)
    _reject_bad_name(name)
    _reject_unusable_timeout(name, timeout_seconds)
    _reject_hold(name, execution_mode)
    hints = _hints(name, target)
    _reject_unusable_return(name, hints)
    description, descriptions = parse_docstring(inspect.getdoc(target))
    if not description:
        raise ConfigurationError(
            f"tool `{name}`: no description. The first paragraph of the docstring "
            f"is what the model reads to decide whether to call this tool."
        )
    parameters = derive_schema(target, descriptions=descriptions)
    return Tool(
        ToolSpec(
            name=name,
            description=description,
            parameters=parameters,
            timeout_seconds=timeout_seconds,
            execution_mode=execution_mode,
            response_instructions=response_instructions,
            http=http,
            dtmf_collected_arguments=dtmf_collected_arguments,
            context_parameter=context_parameter(target, hints),
            is_async=inspect.iscoroutinefunction(target),
            target=target,
        )
    )


def context_parameter(target: Callable, hints: Optional[dict] = None) -> Optional[str]:
    # By annotation, never by name. The runtime calls a handler as
    # `handler(**arguments)`, so a context passed unconditionally would break
    # every handler that already exists; and the name is still needed here,
    # because injection has to supply it as a keyword.
    resolved = hints if hints is not None else get_type_hints(target)
    for name in inspect.signature(target).parameters:
        if resolved.get(name) is ToolContext:
            return name
    return None


def parse_docstring(doc: Optional[str]) -> tuple:
    lines = (doc or "").splitlines()
    index = 0
    body = []
    while index < len(lines) and not _SECTION.match(lines[index]):
        body.append(lines[index])
        index += 1
    descriptions: dict = {}
    while index < len(lines):
        if _ARGS_SECTION.match(lines[index]):
            index = _read_args(lines, index + 1, descriptions)
            continue
        index += 1
    return _first_paragraph(body), descriptions


def _first_paragraph(lines: list) -> str:
    paragraph: list = []
    for line in lines:
        if not line.strip():
            if paragraph:
                break
            continue
        paragraph.append(line.strip())
    return " ".join(paragraph)


def _read_args(lines: list, index: int, descriptions: dict) -> int:
    name = None
    while index < len(lines):
        line = lines[index]
        # A section header sits at the docstring's own indent, so an indented line
        # never closes the block it belongs to.
        if _SECTION.match(line):
            return index
        stripped = line.strip()
        if not stripped:
            index += 1
            continue
        match = _ARG.match(stripped)
        if match is not None:
            name = match.group(1)
            descriptions[name] = match.group(2).strip()
        elif name is not None:
            descriptions[name] = f"{descriptions[name]} {stripped}".strip()
        index += 1
    return index


def _hints(name: str, target: Callable) -> dict:
    try:
        return get_type_hints(target)
    except NameError as exc:
        raise ConfigurationError(
            f"tool `{name}`: an annotation names something that does not resolve "
            f"({exc}), so no schema can be derived from it."
        ) from exc


def _reject_unusable_return(name: str, hints: dict) -> None:
    if "return" not in hints:
        raise ConfigurationError(
            f"tool `{name}`: no return annotation. The result is serialised for "
            f"the model, so its type has to be stated."
        )
    annotation = hints["return"]
    if _is_json_serialisable(annotation):
        return
    raise ConfigurationError(
        f"tool `{name}`: return type `{_render(annotation)}` is not "
        f"JSON-serialisable. Return one of: {_JSON_RETURNS}."
    )


def _is_json_serialisable(annotation: Any) -> bool:
    if annotation is None or annotation is type(None):
        return True
    if annotation in (str, int, float, bool, dict, list):
        return True
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return True
    origin = get_origin(annotation)
    # Not recursed into: a result reaches the model through `str(result)`
    # (`connection.py`, `realtime.py`), never `json.dumps`, so a container of
    # an unserialisable element has nothing to fail at.
    if origin in (dict, list):
        return True
    if origin in (Union, UnionType):
        return all(_is_json_serialisable(arg) for arg in get_args(annotation))
    return False


def _reject_platform_name(name: str) -> None:
    if name not in PLATFORM_TOOL_NAMES:
        return
    raise ConfigurationError(
        f"tool `{name}` is the name of an AssemblyAI platform tool. Backend "
        f"selection matches on the name before anything else, so the platform "
        f"implementation would answer every call and this function would never "
        f"run. Rename it."
    )


def _reject_bad_name(name: str) -> None:
    if _SNAKE_CASE.match(name):
        return
    raise ConfigurationError(f"tool `{name}`: a tool name must be snake_case.")


def _reject_unusable_timeout(name: str, timeout_seconds: Any) -> None:
    if (
        isinstance(timeout_seconds, int)
        and MIN_TIMEOUT_SECONDS <= timeout_seconds <= MAX_TIMEOUT_SECONDS
    ):
        return
    raise ConfigurationError(
        f"tool `{name}`: timeout_seconds={timeout_seconds!r} is outside "
        f"{MIN_TIMEOUT_SECONDS}-{MAX_TIMEOUT_SECONDS} seconds, which is the range "
        f"the agent row accepts, so the deploy would be rejected."
    )


def _reject_hold(name: str, execution_mode: Any) -> None:
    if getattr(execution_mode, "value", execution_mode) != "hold":
        return
    raise ConfigurationError(
        f"tool `{name}`: execution_mode `hold` is not in v1. It suppresses caller "
        f"transcripts and filler audio while the tool runs, which needs a warning "
        f"written before it is offered. Use `interactive`."
    )
