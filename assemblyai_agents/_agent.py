import re
import textwrap
from dataclasses import dataclass
from typing import Any, Optional

from ._exceptions import ConfigurationError
from ._io import AudioInput, AudioOutput
from ._telephony import (
    HumanTransfer,
    PreConnectRequest,
    require_e164,
    require_trunk_for_transfers,
    validate_pre_connect,
)
from ._tool import Tool
from .models.rest import (
    AgentCreateRequest,
    AgentUpdateRequest,
    LlmConfigRequest,
    PlaintextPreConnectRequest,
    PlaintextToolDefinition,
    TransferTarget,
    VoiceConfig,
)

_REASONING_EFFORT_RE = re.compile(r"[a-z][a-z0-9_-]{0,31}")

# The field is read from inside the stored input's turn-detection block. The key
# there is spelled exactly like the `VoiceAgent` field, so there is one name to
# know rather than one to translate.
TURN_DETECTION_KEY = "turn_detection"
GREETING_UNINTERRUPTIBLE_KEY = "greeting_uninterruptible_seconds"
MAX_GREETING_UNINTERRUPTIBLE_SECONDS = 30.0


@dataclass(frozen=True, kw_only=True)
class VoiceAgent:
    """A declaration of one agent, and a builder for the request that creates it.

    Every field here is a field of ``AgentCreateRequest``; ``to_request()``
    returns that model, so what the SDK sends is inspectable and assertable
    without a network call. Three normalisations are the only shape difference:
    ``voice`` is a name here and a nested ``VoiceConfig`` on the wire; ``llm``
    takes one config here and goes out as the one-element list the wire field
    expects (it is capped at one in v1); and ``input``/``output`` are typed
    helpers here and plain dicts on the wire.

    ``llm`` left unset is itself a declaration: the agent runs on the model
    endpoint AssemblyAI supplies, which for a project deployed with
    ``--type service`` is that deployment's own. Nothing is sent — the key is
    absent from the request rather than ``null`` or ``[]`` — so there is no
    address to copy and no key to invent, and nothing to repoint when the
    deployment moves. Set ``llm`` only to point the agent at an endpoint you
    operate yourself; see :mod:`assemblyai_agents.byo`.

    ``keyterms`` is not a field here. It lives inside ``input`` on the wire, and
    a shortcut that relocates a field is the second vocabulary this builder
    exists to remove — write ``input=AudioInput(keyterms=[...])``.

    ``greeting_uninterruptible_seconds`` is the one field that breaks that rule,
    and it sits here rather than on :class:`AudioInput` because it is a property
    of the greeting next to it. It goes out as
    ``input.turn_detection.greeting_uninterruptible_seconds`` — the same name it
    has here, so the block it lands in is the only thing to learn. It is a
    duration
    rather than a flag because an uninterruptible turn *discards* the caller's
    speech instead of queueing it: the caller talks, is talked over, and what
    they said is never answered and never joins the conversation. A duration
    bounds how long that can happen. ``None`` sends nothing and 0.0 is off.

    ``platform_tools_enabled`` is ``True`` by default, which is AssemblyAI's own
    default too. Set it ``False`` when your model endpoint cannot return tool
    calls: nothing of ours is then added to the tool list the model sees. Only
    one tool is ever added at you — ``transfer_call``, and only when
    ``transfer_targets`` is configured — so turning this off costs you human
    transfer and nothing else. The server refuses a declaration that turns it
    off and still names a platform tool in ``tools``.

    ``transfer_targets``, ``pre_connect``, ``outbound_trunk_id`` and
    ``caller_id`` are telephony fields and inert on a WebSocket session. Every
    rule about them that is decidable without the network is checked here rather
    than on the round trip: see :class:`HumanTransfer` and
    :class:`PreConnectRequest` for the traps each carries. Two rules are not
    decidable here and are left to the server — whether a ``caller_id`` belongs
    to the account, and whether a target agent exists.
    """

    name: str
    system_prompt: str
    voice: str
    greeting: Optional[str] = None
    greeting_uninterruptible_seconds: Optional[float] = None
    llm: Optional[LlmConfigRequest] = None
    input: Optional[AudioInput] = None
    output: Optional[AudioOutput] = None
    tools: Optional[list[Tool]] = None
    transfer_targets: Optional[list[HumanTransfer]] = None
    pre_connect: Optional[list[PreConnectRequest]] = None
    outbound_trunk_id: Optional[str] = None
    caller_id: Optional[str] = None
    platform_tools_enabled: bool = True

    def __init_subclass__(cls, **kwargs) -> None:
        raise ConfigurationError(
            f"class `{cls.__name__}` subclasses VoiceAgent. A declaration is data, "
            f"not a base class: inheritance would reopen the surface this design "
            f"closes. Write `{cls.__name__.lower()} = VoiceAgent(...)` instead."
        )

    def __post_init__(self) -> None:
        # A prompt is written as an indented triple-quoted block, so the indent is
        # an artifact of where it sits in the file rather than something the model
        # should read. Frozen, so the rewrite goes in behind the generated setattr.
        object.__setattr__(
            self, "system_prompt", textwrap.dedent(self.system_prompt).strip()
        )
        _require_unique_tool_names(self.tools)
        _require_base_url_with_key(self.llm)
        _require_reasoning_effort_format(self.llm)
        _require_usable_greeting_protection(
            self.greeting_uninterruptible_seconds, self.greeting, self.input
        )
        validate_pre_connect(self.pre_connect)
        require_trunk_for_transfers(self.transfer_targets, self.outbound_trunk_id)
        if self.caller_id is not None:
            require_e164("caller_id", self.caller_id)

    def to_request(self) -> AgentCreateRequest:
        return AgentCreateRequest(
            name=self.name,
            system_prompt=self.system_prompt,
            greeting=self.greeting,
            voice=VoiceConfig(voice_id=self.voice),
            input=self.input_config(),
            output=self.output.to_dict() if self.output is not None else None,
            tools=self.tool_definitions(),
            pre_connect_requests=self.pre_connect_requests(),
            transfer_targets=self.wire_transfer_targets(),
            outbound_trunk_id=self.outbound_trunk_id,
            caller_id=self.caller_id,
            llm=None if self.llm is None else [self.llm],
            platform_tools_enabled=self.platform_tools_enabled,
        )

    def to_update_request(self) -> AgentUpdateRequest:
        """The same declaration, as the model ``PUT /v1/agents/{id}`` takes.

        Every field is sent, because the endpoint replaces the stored agent
        rather than merging into it: anything left out is dropped from the
        stored row, not preserved.
        """
        # Field by field off the create model rather than through a `model_dump`
        # round trip: re-validating a dumped payload would refill every default
        # the create model deliberately left unset, `execution_mode` included.
        created = self.to_request()
        fields = {
            name: getattr(created, name) for name in AgentCreateRequest.model_fields
        }
        if self.llm is not None and self.llm.reasoning_effort is None:
            # The server keeps an omitted reasoning_effort, and the payload drops
            # None, so "" is the only way this declaration can say "unset".
            fields["llm"] = [self.llm.model_copy(update={"reasoning_effort": ""})]
        return AgentUpdateRequest(**fields)

    def input_config(self) -> Optional[dict[str, Any]]:
        """The ``input`` block as the wire carries it.

        ``greeting_uninterruptible_seconds`` is folded in here, because the
        server reads it from the turn-detection block rather than from a
        top-level field. An agent that sets it without an ``input`` still gets
        one, since there is nowhere else for the key to go.
        """
        emitted = self.input.to_dict() if self.input is not None else None
        if self.greeting_uninterruptible_seconds is None:
            return emitted
        if emitted is None:
            emitted = AudioInput().to_dict()
        turn_detection = dict(emitted.get(TURN_DETECTION_KEY) or {})
        turn_detection[GREETING_UNINTERRUPTIBLE_KEY] = (
            self.greeting_uninterruptible_seconds
        )
        return {**emitted, TURN_DETECTION_KEY: turn_detection}

    def tool_definitions(self) -> Optional[list[PlaintextToolDefinition]]:
        if self.tools is None:
            return None
        return [declared.definition() for declared in self.tools]

    def pre_connect_requests(self) -> Optional[list[PlaintextPreConnectRequest]]:
        if self.pre_connect is None:
            return None
        return [entry.to_request() for entry in self.pre_connect]

    def wire_transfer_targets(self) -> Optional[list[TransferTarget]]:
        if self.transfer_targets is None:
            return None
        return [target.to_target() for target in self.transfer_targets]

    def client_resident_tool_names(self) -> tuple[str, ...]:
        """Names of the tools this process has to answer over the WebSocket.

        A tool with no ``http`` config and no platform-catalog entry is resolved
        by the connected client, and
        only the WebSocket transport can do that.
        """
        return tuple(
            declared.name for declared in self.tools or () if declared.spec.http is None
        )


def _require_usable_greeting_protection(
    seconds: Optional[float],
    greeting: Optional[str],
    audio_input: Optional[AudioInput],
) -> None:
    """Three rules, all decidable here, none worth a round trip to learn.

    The range mirrors the server's own bound. The other two catch a value that
    would be accepted and then do nothing: a protected greeting that is never
    spoken, and a second spelling of the same key under ``extra``.
    """
    if seconds is None:
        return
    if not 0.0 <= seconds <= MAX_GREETING_UNINTERRUPTIBLE_SECONDS:
        raise ConfigurationError(
            f"greeting_uninterruptible_seconds={seconds!r} is outside "
            f"0.0-{MAX_GREETING_UNINTERRUPTIBLE_SECONDS}, which is the range the "
            f"agent row accepts, so the deploy would be rejected. The upper bound "
            f"is there because the caller's speech is discarded, not queued, for "
            f"as long as the greeting is protected. 0.0 turns it off."
        )
    if greeting is None:
        raise ConfigurationError(
            "greeting_uninterruptible_seconds is set but `greeting` is not. The "
            "window protects the greeting, so with no greeting to speak there is "
            "nothing to protect and the setting would be accepted and ignored. "
            "Set `greeting`, or drop this field."
        )
    extra = (audio_input.extra if audio_input is not None else None) or {}
    collides = GREETING_UNINTERRUPTIBLE_KEY in (extra.get(TURN_DETECTION_KEY) or {})
    if collides:
        raise ConfigurationError(
            f"greeting_uninterruptible_seconds is set and "
            f"input.extra['{TURN_DETECTION_KEY}'] also sets "
            f"`{GREETING_UNINTERRUPTIBLE_KEY}`. That is the same key written "
            f"twice, and the typed field wins silently. Keep one of them."
        )


def _require_base_url_with_key(llm: Optional[LlmConfigRequest]) -> None:
    """A key with no address is a key handed to nobody, and the server says so.

    Saying nothing about the model is a declaration in its own right: the agent
    runs on the endpoint AssemblyAI supplies from its own deployment. Naming
    only a model is the same declaration with a model picked. Naming only a key
    is neither — it is a half-deleted bring-your-own config, and the server
    refuses it.
    """
    if llm is None or llm.base_url or not llm.api_key:
        return
    raise ConfigurationError(
        "llm has an api_key but no base_url. A key with no address to send it "
        "to cannot be used, and the server refuses the pair. Give the "
        "endpoint's base_url, or leave `llm` unset, which is how an agent says "
        "it runs on the model endpoint AssemblyAI supplies from its own "
        "deployment."
    )


def _require_reasoning_effort_format(llm: Optional[LlmConfigRequest]) -> None:
    effort = None if llm is None else llm.reasoning_effort
    if not effort or _REASONING_EFFORT_RE.fullmatch(effort):
        return
    raise ConfigurationError(
        f"llm.reasoning_effort {effort!r} must be one word of up to 32 lowercase "
        "letters, digits, '_' or '-', starting with a letter, such as 'none', "
        "'low', 'medium' or 'high'. The server refuses anything else."
    )


def _require_unique_tool_names(tools: Optional[list[Tool]]) -> None:
    # The server refuses this too, on create and on update. Refusing it here
    # turns a round trip into an immediate error.
    seen: set = set()
    for declared in tools or ():
        if declared.name in seen:
            raise ConfigurationError(
                f"tool `{declared.name}` is listed twice. A name identifies one "
                f"tool, so the second would replace the first and the agent would "
                f"deploy without it."
            )
        seen.add(declared.name)
