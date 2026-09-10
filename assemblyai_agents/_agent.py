import textwrap
from dataclasses import dataclass
from typing import Optional

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

    ``keyterms`` is not a field here. It lives inside ``input`` on the wire, and
    a shortcut that relocates a field is the second vocabulary this builder
    exists to remove — write ``input=AudioInput(keyterms=[...])``.

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
    llm: Optional[LlmConfigRequest] = None
    input: Optional[AudioInput] = None
    output: Optional[AudioOutput] = None
    tools: Optional[list[Tool]] = None
    transfer_targets: Optional[list[HumanTransfer]] = None
    pre_connect: Optional[list[PreConnectRequest]] = None
    outbound_trunk_id: Optional[str] = None
    caller_id: Optional[str] = None

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
            input=self.input.to_dict() if self.input is not None else None,
            output=self.output.to_dict() if self.output is not None else None,
            tools=self.tool_definitions(),
            pre_connect_requests=self.pre_connect_requests(),
            transfer_targets=self.wire_transfer_targets(),
            outbound_trunk_id=self.outbound_trunk_id,
            caller_id=self.caller_id,
            llm=None if self.llm is None else [self.llm],
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
        return AgentUpdateRequest(
            **{name: getattr(created, name) for name in AgentCreateRequest.model_fields}
        )

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
