import re
from dataclasses import dataclass
from typing import Literal, Optional

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
ON_FAILURE = ("continue", "reject")

MIN_RING_TIMEOUT = 1
MAX_RING_TIMEOUT = 600
MIN_PRE_CONNECT_TIMEOUT_MS = 1
MAX_PRE_CONNECT_TIMEOUT_MS = 800
MAX_PRE_CONNECT_REQUESTS = 2

# Facts about the call itself that the platform supplies. Any entry may name
# these in `sends` without an earlier entry having captured them. They are not
# captures, so `validate_pre_connect` keeps them out of the set it checks for
# duplicate `returns` names.
CALL_FACTS = frozenset(
    {"caller_number", "dialed_number", "direction", "agent_id", "session_id"}
)

# The only member of `allow_overrides`'s closed vocabulary, which is why the
# field is a flag here rather than a list.
GREETING_OVERRIDE = "greeting"

_E164 = re.compile(r"^\+[1-9]\d{1,14}$")

TransferMode = Literal["cold", "warm"]
Method = Literal["GET", "POST", "PUT", "PATCH", "DELETE"]
OnFailure = Literal["continue", "reject"]


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
    """One HTTPS call made before the conversation starts.

    Four behaviours are not in the wire model and will surprise anyone who
    assumes otherwise.

    **Pre-connect fails open unless the entry asks to be a gate.**
    By default a timeout, a 500, an unparseable body costs this entry's values
    and nothing more: the call proceeds without them. ``on_failure="reject"``
    refuses the call instead, on every kind of failure — a timeout, a non-2xx,
    a DNS failure, a connect failure — with no special case for any of them.
    It exists because a pre-connect request can override the voice and the
    greeting. If the request that picks the voice fails and the call is
    answered anyway with the declared defaults, it picks up as the wrong
    persona, and a customer may hold that to be worse than not answering.

    **A greeting override is read from a top-level ``greeting`` key in the
    response**, not from a value named in ``returns``.
    Setting ``allow_overrides=True`` permits it; naming a capture ``greeting``
    does not deliver it.

    **A top-level ``reject: true`` in the response aborts the call.** That is
    the one thing a pre-connect endpoint can do to stop a conversation.

    **The platform's own call facts are sent only when named.**
    ``caller_number``, ``dialed_number``, ``direction`` (``"inbound"`` or
    ``"outbound"``), ``agent_id`` and ``session_id`` need no earlier capture,
    but they are opt-in per request: an entry that names none of them sends
    nothing. A fact the platform does not have — empty, or a carrier
    placeholder such as ``anonymous`` — is left out of the payload rather than
    sent blank, so the ``default`` declared for that name applies instead. A
    name an earlier entry captured wins over the platform's value, so a
    ``Captured`` carrying a ``default`` is how to guarantee the key is always
    present.

    ``allow_overrides`` is a flag rather than a list because the wire field's
    vocabulary is closed and holds one value, ``greeting``.
    """

    url: str
    method: Method = "POST"
    headers: Optional[list[Header]] = None
    sends: Optional[list[str]] = None
    returns: Optional[list[Captured]] = None
    timeout_ms: Optional[int] = None
    allow_overrides: bool = False
    on_failure: OnFailure = "continue"

    def __post_init__(self) -> None:
        if not self.url.startswith("https://"):
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
        if self.on_failure not in ON_FAILURE:
            raise ConfigurationError(
                f"pre-connect url={self.url!r}: on_failure={self.on_failure!r} is not "
                f"`continue` or `reject`. `continue` proceeds without this request's "
                f"values; `reject` refuses the call, on any failure."
            )
        _require_unique_capture_names(self.captured_names())

    def captured_names(self) -> tuple[str, ...]:
        return tuple(captured.name for captured in self.returns or ())

    def to_request(self) -> PlaintextPreConnectRequest:
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
            on_failure=self.on_failure,
        )


def validate_pre_connect(entries: Optional[list[PreConnectRequest]]) -> None:
    if not entries:
        return
    if len(entries) > MAX_PRE_CONNECT_REQUESTS:
        raise ConfigurationError(
            f"{len(entries)} pre-connect requests are declared; the API accepts at "
            f"most {MAX_PRE_CONNECT_REQUESTS}."
        )
    # `resolvable` tracks captures alone; `sendable` additionally admits the
    # platform's call facts. Keeping them apart is what lets a capture be named
    # after a call fact without reading as a duplicate — the server splits the
    # two sets the same way, for the same reason.
    resolvable: set = set()
    for entry in entries:
        sendable = CALL_FACTS | resolvable
        # A platform call fact is sendable anywhere. Anything else is only
        # sendable once an *earlier* entry has captured it, because entries
        # run in order. The server enforces the same ordering rule.
        for name in entry.sends or ():
            if name in sendable:
                continue
            raise ConfigurationError(
                f"pre-connect url={entry.url!r} sends `{name}`, which is neither a "
                f"call fact the platform supplies nor a name an earlier entry "
                f"captures. Entries run in order, so only a name returned by an "
                f"earlier one is resolvable. The call facts, which any entry may "
                f"send, are {', '.join(sorted(CALL_FACTS))}."
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
