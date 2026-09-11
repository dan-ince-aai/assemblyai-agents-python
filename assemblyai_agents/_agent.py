import re
import textwrap
import warnings
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Optional

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
    HttpToolHeaderInput,
    LlmConfigRequest,
    PlaintextPreConnectRequest,
    PlaintextToolDefinition,
    TransferTarget,
    VoiceConfig,
)


@dataclass(frozen=True, kw_only=True)
class VoiceAgent:
    """A declaration of one agent, and a builder for the request that creates it.

    Two things decide what the agent is. ``system_prompt`` shapes what the
    platform's own model says. ``reply`` replaces that model with your code:
    the platform asks your function what to say on every turn. One or the
    other; there is no third mode.

    Everything the platform has to reach — the tools, a pre-connect handler,
    the reply endpoint — is a URL on the wire, and every one of those URLs is
    *this process* plus a path. So the declaration does not carry them. It
    carries the functions, and the address is supplied once, either as
    ``public_url`` here or to ``serve()`` / ``deploy()``, which bind it:

        agent = VoiceAgent(name="…", voice="alba", system_prompt="…",
                           tools=[order_status], reply=decide)
        agent.serve()        # PUBLIC_BASE_URL → deploy → serve

    Every field maps onto ``AgentCreateRequest``, and ``to_request()`` returns
    that model, so what the SDK sends is inspectable without a network call.
    Three normalisations are the only shape difference: ``voice`` is a name
    here and a nested ``VoiceConfig`` on the wire; ``reply`` becomes the
    one-element ``llm`` list pointing at ``{public_url}/v1``; ``input`` /
    ``output`` are typed helpers here and plain dicts on the wire.

    ``transfer_targets``, ``pre_connect``, ``outbound_trunk_id`` and
    ``caller_id`` are telephony fields and inert on a WebSocket session.
    """

    name: str
    system_prompt: str
    voice: str
    greeting: Optional[str] = None
    reply: Optional[Callable[..., Any]] = None
    input: Optional[AudioInput] = None
    output: Optional[AudioOutput] = None
    tools: Optional[list[Tool]] = None
    transfer_targets: Optional[list[HumanTransfer]] = None
    pre_connect: Optional[list[PreConnectRequest]] = None
    outbound_trunk_id: Optional[str] = None
    caller_id: Optional[str] = None

    # The address the platform reaches this process at, and the secret it
    # presents when it does. Usually bound by serve() / deploy(); pass them here
    # when the address is already known, so to_request() works directly.
    public_url: Optional[str] = None
    secret: Optional[str] = field(default=None, repr=False)

    # Deprecated: pointing the agent at a third-party model with no code. Use
    # `reply=` and call whatever model you like from inside it.
    llm: Optional[LlmConfigRequest] = None

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
        if self.reply is not None and not callable(self.reply):
            raise ConfigurationError(
                f"reply={self.reply!r} is not callable. Pass the function that decides "
                f"what the agent says: `reply=decide`, where `decide(turn)` returns "
                f"say(...), call_tool(...) or silence()."
            )
        if self.llm is not None:
            if self.reply is not None:
                raise ConfigurationError(
                    "both `reply=` and `llm=` are set. An agent has two modes: the "
                    "platform's model talks (leave both out), or your code does "
                    "(`reply=`). To use another model, call it from inside `reply`."
                )
            warnings.warn(
                "VoiceAgent(llm=...) is deprecated. Write a `reply=` function and call "
                "any model you like from inside it; the endpoint the platform calls is "
                "then this process, with nothing to configure.",
                DeprecationWarning,
                stacklevel=3,
            )
        if self.public_url is not None:
            url = self.public_url.rstrip("/")
            if not url.startswith("https://"):
                raise ConfigurationError(
                    f"public_url={self.public_url!r} is not https. The platform reaches "
                    f"this process over public HTTPS only."
                )
            object.__setattr__(self, "public_url", url)
        _require_unique_tool_names(self.tools)
        validate_pre_connect(self.pre_connect)
        require_trunk_for_transfers(self.transfer_targets, self.outbound_trunk_id)
        if self.caller_id is not None:
            require_e164("caller_id", self.caller_id)

    # ------------------------------------------------------------------ hosting

    @property
    def hosts_replies(self) -> bool:
        """True when this process decides what the agent says."""
        return self.reply is not None

    def hosted_tool_names(self) -> tuple[str, ...]:
        """Names of the tools this process serves, as opposed to an external URL."""
        return tuple(declared.name for declared in self.tools or () if declared.hosted)

    def hosted_pre_connect_paths(self) -> tuple[str, ...]:
        """Paths of the pre-connect handlers this process serves."""
        return tuple(entry.path for entry in self.pre_connect or () if entry.handler is not None)

    @property
    def needs_address(self) -> bool:
        """Does anything on this declaration point back at this process?"""
        return bool(
            self.hosts_replies or self.hosted_tool_names() or self.hosted_pre_connect_paths()
        )

    def hosted_at(self, public_url: str, *, secret: Optional[str] = None) -> "VoiceAgent":
        """The same declaration, bound to the address the platform reaches it at.

        Fills in every URL the wire needs from one base: each hosted tool at
        ``{public_url}/tools/{name}``, each pre-connect handler at its path,
        and — when ``reply`` is set — the reply endpoint at ``{public_url}/v1``.
        ``secret`` becomes the bearer the platform presents on all of them.

        Returns a new declaration; this one is untouched.
        """
        return replace(self, public_url=public_url, secret=secret)

    def _require_address(self, verb: str) -> str:
        if self.public_url:
            return self.public_url
        raise ConfigurationError(
            f"agent `{self.name}` has {_hosted_summary(self)}, so the platform needs an "
            f"address to reach this process, and none is set. Either pass "
            f"`public_url=` (an https address that reaches this process), set "
            f"PUBLIC_BASE_URL and call `agent.{verb}()`, or bind one with "
            f"`agent.hosted_at(url, secret=...)`."
        )

    def _auth_headers(self) -> Optional[list[HttpToolHeaderInput]]:
        if not self.secret:
            return None
        return [HttpToolHeaderInput(name="Authorization", value=f"Bearer {self.secret}")]

    def _bound_tools(self) -> Optional[list[Tool]]:
        if self.tools is None:
            return None
        if not self.hosted_tool_names():
            return list(self.tools)
        base = self._require_address("deploy")
        return [
            declared.hosted_at(f"{base}/tools/{declared.name}", headers=self._auth_headers())
            if declared.hosted
            else declared
            for declared in self.tools
        ]

    def _bound_pre_connect(self) -> Optional[list[PreConnectRequest]]:
        if self.pre_connect is None:
            return None
        if not self.hosted_pre_connect_paths():
            return list(self.pre_connect)
        base = self._require_address("deploy")
        return [
            entry.hosted_at(base, secret=self.secret) if entry.handler is not None else entry
            for entry in self.pre_connect
        ]

    def _wire_llm(self) -> Optional[list[LlmConfigRequest]]:
        if self.llm is not None:
            return [self.llm]
        if self.reply is None:
            return None
        base = self._require_address("deploy")
        return [
            LlmConfigRequest(
                base_url=f"{base}/v1",
                model=_slug(self.name),
                # Write-only on the platform; what serve() checks on the reply route.
                api_key=self.secret or "",
            )
        ]

    # ------------------------------------------------------------------ the wire

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
            llm=self._wire_llm(),
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
        bound = self._bound_tools()
        if bound is None:
            return None
        return [declared.definition() for declared in bound]

    def pre_connect_requests(self) -> Optional[list[PlaintextPreConnectRequest]]:
        bound = self._bound_pre_connect()
        if bound is None:
            return None
        return [entry.to_request() for entry in bound]

    def wire_transfer_targets(self) -> Optional[list[TransferTarget]]:
        if self.transfer_targets is None:
            return None
        return [target.to_target() for target in self.transfer_targets]

    # ------------------------------------------------------------------ running it

    def deploy(self, **kwargs: Any) -> str:
        """Create the agent, or update the one whose id is stored. Returns the id.

        See :func:`assemblyai_agents.deploy.deploy` for the keyword arguments.
        """
        from .deploy import deploy

        return deploy(self, **kwargs)

    def serve(self, **kwargs: Any) -> Any:
        """Address → deploy → serve, in one call. Blocks until interrupted.

        See :func:`assemblyai_agents.serving.serve_agent` for the keyword
        arguments. The address comes from ``public_url=`` or ``PUBLIC_BASE_URL``;
        nothing here starts a tunnel.
        """
        from .serving import serve_agent

        return serve_agent(self, **kwargs)


def _hosted_summary(agent: VoiceAgent) -> str:
    parts = []
    if agent.hosts_replies:
        parts.append("a `reply` function")
    names = agent.hosted_tool_names()
    if names:
        parts.append(f"hosted tools ({', '.join(f'`{n}`' for n in names)})")
    if agent.hosted_pre_connect_paths():
        parts.append("a pre-connect handler")
    return " and ".join(parts) or "nothing hosted"


def _slug(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "replies"


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
