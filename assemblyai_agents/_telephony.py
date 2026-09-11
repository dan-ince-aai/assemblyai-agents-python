import re
from dataclasses import dataclass, replace
from typing import Any, Callable, Literal, Optional

from ._exceptions import ConfigurationError
from .models.rest import (
    HttpMethod,
    HttpToolHeaderInput,
    PlaintextHttpToolConfig,
    PlaintextPreConnectRequest,
    PreConnectReturn,
    TransferTarget,
)

TRANSFER_MODES = ("cold", "warm")
HTTP_METHODS = ("GET", "POST", "PUT", "PATCH", "DELETE")

MIN_RING_TIMEOUT = 1
MAX_RING_TIMEOUT = 600
MIN_PRE_CONNECT_TIMEOUT_MS = 1
MAX_PRE_CONNECT_TIMEOUT_MS = 800
MAX_PRE_CONNECT_REQUESTS = 2

# The only member of `allow_overrides`'s closed vocabulary, which is why the
# field is a flag here rather than a list.
GREETING_OVERRIDE = "greeting"

_E164 = re.compile(r"^\+[1-9]\d{1,14}$")

TransferMode = Literal["cold", "warm"]
Method = Literal["GET", "POST", "PUT", "PATCH", "DELETE"]


@dataclass(frozen=True, kw_only=True)
class HumanTransfer:
    """One phone number a live call may be handed to.

    Three things are worth knowing before using this.

    Transfer targets are **telephony-only**. A WebSocket session ignores them
    outright, so declaring one changes nothing about a WebSocket call.

    A human target **requires the agent's ``outbound_trunk_id``**, which is
    checked when the agent is built. That trunk affects human transfers *only*,
    despite its own field description on the REST API: an outbound call placed
    through the calls API uses the platform's own trunk instead.

    On a **cold** transfer the server silently ignores ``consult_instructions``
    and ``consult_timeout`` and rejects ``record_consult``. All three are
    refused here for a cold target rather than reproducing that split.

    ``kind="agent"`` is not offered in v1. It is validated at create, but the
    transfer coordinator never reads it and the call ends in "[misconfigured]".
    """

    name: str
    phone_number: str
    mode: TransferMode = "cold"
    ring_timeout: Optional[int] = None
    consult_instructions: Optional[str] = None
    consult_timeout: Optional[int] = None
    record_consult: Optional[bool] = None

    def __post_init__(self) -> None:
        require_e164(f"transfer target `{self.name}`.phone_number", self.phone_number)
        if self.mode not in TRANSFER_MODES:
            raise ConfigurationError(
                f"transfer target `{self.name}`: mode={self.mode!r} is not `cold` or "
                f"`warm`."
            )
        _require_in_range(
            f"transfer target `{self.name}`.ring_timeout",
            self.ring_timeout,
            MIN_RING_TIMEOUT,
            MAX_RING_TIMEOUT,
        )
        if self.mode == "warm":
            return
        for field in ("consult_instructions", "consult_timeout", "record_consult"):
            if getattr(self, field) is None:
                continue
            raise ConfigurationError(
                f"transfer target `{self.name}`: {field} applies to a warm transfer "
                f"only. On a cold transfer the server ignores the consult fields "
                f"and rejects `record_consult`, so this target would not behave as "
                f'written. Set mode="warm" or drop {field}.'
            )

    def to_target(self) -> TransferTarget:
        return TransferTarget(
            name=self.name,
            kind="human",
            phone_number=self.phone_number,
            mode=self.mode,
            ring_timeout_seconds=self.ring_timeout,
            consult_instructions=self.consult_instructions,
            consult_timeout_seconds=self.consult_timeout,
            record_consult=self.record_consult,
        )


@dataclass(frozen=True, kw_only=True)
class Header:
    """One HTTP header sent with a pre-connect request.

    ``value`` is required and there is no name-only form, because the API keeps
    stored header secrets **by list position**: on update they are re-joined to
    the incoming list by index, and the fail-closed guard only trips when the kept header's *name* is absent at that index. Two
    entries that both use a header called ``Authorization`` therefore pass the
    guard and swap credentials when the list is reordered. Sending an explicit
    value every time removes index-joining from anything this SDK sends, which
    is also why ``remove: true`` is not modelled.
    """

    name: str
    value: str


@dataclass(frozen=True, kw_only=True)
class Captured:
    """One value read out of a pre-connect response body.

    ``path`` is a dotted path — ``customer.tier``, ``results.0.id``. Its shape
    is checked at write time; whether it resolves is only knowable on the call,
    and an unresolved path with no ``default`` leaves the value unset.
    """

    name: str
    path: str
    default: Optional[str] = None


@dataclass(frozen=True, kw_only=True)
class PreConnectRequest:
    """One HTTPS call made before a phone call is answered.

    Give it a ``handler`` and this process serves it: ``serve()`` answers
    ``POST /pre-connect/{handler name}`` with your function, and the deploy
    binds that address. Give it a ``url`` instead to have the platform call a
    service you already run. One or the other.

        PreConnectRequest(handler=lookup,
                          returns=[Captured(name="tier", path="customer.tier")],
                          allow_overrides=True)

    Three behaviours are not in the wire model and will surprise anyone who
    assumes otherwise.

    **Pre-connect fails open on every error.**
    A timeout, a 500, an unparseable body: the call proceeds without the values.
    It is not a gate.

    **A greeting override is read from a top-level ``greeting`` key in the
    response**, not from a value named in ``returns``.
    Setting ``allow_overrides=True`` permits it; naming a capture ``greeting``
    does not deliver it.

    **A top-level ``reject: true`` in the response aborts the call.** That is
    the one thing a pre-connect endpoint can do to stop a conversation.

    ``allow_overrides`` is a flag rather than a list because the wire field's
    vocabulary is closed and holds one value, ``greeting``.
    """

    url: Optional[str] = None
    handler: Optional[Callable[[dict], Any]] = None
    method: Method = "POST"
    headers: Optional[list[Header]] = None
    sends: Optional[list[str]] = None
    returns: Optional[list[Captured]] = None
    timeout_ms: Optional[int] = None
    allow_overrides: bool = False

    def __post_init__(self) -> None:
        if self.url is None and self.handler is None:
            raise ConfigurationError(
                "a pre-connect request needs `handler=` (a function this process serves) "
                "or `url=` (a service you already run). Binding a handler to an address "
                "sets both, which is fine."
            )
        if self.handler is not None and not callable(self.handler):
            raise ConfigurationError(
                f"pre-connect handler={self.handler!r} is not callable. Pass a function "
                f"taking the request body as a dict and returning a dict."
            )
        if self.url is not None and not self.url.startswith("https://"):
            raise ConfigurationError(
                f"pre-connect url={self.url!r} is not https. The API accepts https "
                f"endpoints only."
            )
        if self.method not in HTTP_METHODS:
            raise ConfigurationError(
                f"pre-connect url={self.url!r}: method={self.method!r} is not one of "
                f"{', '.join(HTTP_METHODS)}."
            )
        _require_in_range(
            f"pre-connect url={self.url!r} timeout_ms",
            self.timeout_ms,
            MIN_PRE_CONNECT_TIMEOUT_MS,
            MAX_PRE_CONNECT_TIMEOUT_MS,
        )
        if not isinstance(self.allow_overrides, bool):
            raise ConfigurationError(
                f"pre-connect url={self.url!r}: allow_overrides is a flag, not a "
                f"list. The wire field's vocabulary holds one value, `greeting`, so "
                f"`True` is the whole of it."
            )
        _require_unique_capture_names(self.captured_names())

    def captured_names(self) -> tuple[str, ...]:
        return tuple(captured.name for captured in self.returns or ())

    @property
    def path(self) -> str:
        """The route this process serves the handler at."""
        if self.handler is None:
            raise ConfigurationError("this pre-connect request has a url, not a handler.")
        return f"/pre-connect/{self.handler.__name__}"

    def hosted_at(self, public_url: str, *, secret: Optional[str] = None) -> "PreConnectRequest":
        """The same request, with its handler's URL bound to this address."""
        if self.handler is None:
            return self
        headers = list(self.headers or [])
        if secret and not any(h.name.lower() == "authorization" for h in headers):
            headers.insert(0, Header(name="Authorization", value=f"Bearer {secret}"))
        return replace(
            self, url=f"{public_url.rstrip('/')}{self.path}", headers=headers or None
        )

    def to_request(self) -> PlaintextPreConnectRequest:
        if self.url is None:
            raise ConfigurationError(
                f"pre-connect handler `{self.handler.__name__}` has no address yet. Bind "
                f"the declaration first: `agent.hosted_at(public_url, secret=...)`, or "
                f"pass `public_url=` to VoiceAgent, or deploy through `agent.deploy()`."
            )
        return PlaintextPreConnectRequest(
            http=PlaintextHttpToolConfig(
                url=self.url,
                http_method=HttpMethod(self.method),
                headers=None
                if self.headers is None
                else [
                    HttpToolHeaderInput(name=header.name, value=header.value)
                    for header in self.headers
                ],
            ),
            sends=self.sends,
            returns=None
            if self.returns is None
            else [
                PreConnectReturn(
                    name=captured.name,
                    path=captured.path,
                    default=captured.default,
                )
                for captured in self.returns
            ],
            timeout_ms=self.timeout_ms,
            allow_overrides=[GREETING_OVERRIDE] if self.allow_overrides else None,
        )


def validate_pre_connect(entries: Optional[list[PreConnectRequest]]) -> None:
    if not entries:
        return
    if len(entries) > MAX_PRE_CONNECT_REQUESTS:
        raise ConfigurationError(
            f"{len(entries)} pre-connect requests are declared; the API accepts at "
            f"most {MAX_PRE_CONNECT_REQUESTS}."
        )
    resolvable: set = set()
    for entry in entries:
        # Entries run in order, so a name is only resolvable once an *earlier*
        # entry has captured it. The server enforces the same ordering rule.
        for name in entry.sends or ():
            if name in resolvable:
                continue
            raise ConfigurationError(
                f"pre-connect url={entry.url!r} sends `{name}`, which no earlier "
                f"entry captures. Entries run in order, so only a name returned by "
                f"an earlier one is resolvable."
            )
        for name in entry.captured_names():
            if name in resolvable:
                raise ConfigurationError(
                    f"pre-connect capture `{name}` is declared twice. Capture names "
                    f"are global across entries, so the second would shadow the "
                    f"first."
                )
            resolvable.add(name)


def require_trunk_for_transfers(
    targets: Optional[list[HumanTransfer]], outbound_trunk_id: Optional[str]
) -> None:
    if not targets or outbound_trunk_id is not None:
        return
    named = ", ".join(f"`{target.name}`" for target in targets)
    raise ConfigurationError(
        f"transfer targets {named} need an outbound_trunk_id: a human transfer "
        f"dials out over that trunk, and without one the transfer cannot be "
        f"placed."
    )


def require_e164(field: str, number: str) -> None:
    if _E164.match(number):
        return
    raise ConfigurationError(
        f"{field}={number!r} is not E.164. Write it as a `+`, a country code and "
        f"the national number, digits only — `+14155550123`."
    )


def _require_in_range(field: str, value: Optional[int], low: int, high: int) -> None:
    if value is None or low <= value <= high:
        return
    raise ConfigurationError(f"{field}={value!r} is outside {low}-{high}.")


def _require_unique_capture_names(names: tuple) -> None:
    seen: set = set()
    for name in names:
        if name in seen:
            raise ConfigurationError(
                f"pre-connect capture `{name}` is declared twice on one entry. A "
                f"name identifies one captured value."
            )
        seen.add(name)
