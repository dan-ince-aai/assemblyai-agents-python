import json as _json
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import httpx
import pytest
from assemblyai_agents import AsyncClient, Client

API_KEY = "k-test-fixture"


@dataclass
class Recorder:
    """Captures every outgoing httpx.Request the transport sees.

    One Recorder instance is shared by a (sync or async) mock transport and the
    test, so the test can assert headers/method/path/query/body across every
    retry attempt of a single logical request.
    """

    requests: list[httpx.Request] = field(default_factory=list)

    def record(self, request: httpx.Request) -> None:
        self.requests.append(request)

    @property
    def count(self) -> int:
        return len(self.requests)

    def header(self, name: str, attempt: int = -1) -> Optional[str]:
        return self.requests[attempt].headers.get(name)

    def headers(self, name: str) -> list[Optional[str]]:
        return [r.headers.get(name) for r in self.requests]


def _to_response(spec: Any) -> httpx.Response:
    """Normalise a test's page/error spec into an httpx.Response.

    A spec may be an httpx.Response (used verbatim) or a
    (status_code, body, headers) tuple where body is a dict/list (JSON-encoded),
    bytes (sent verbatim — for the non-JSON edge cases), or None (empty body).
    """
    if isinstance(spec, httpx.Response):
        return spec
    if isinstance(spec, (dict, list)):
        return httpx.Response(
            200,
            content=_json.dumps(spec).encode("utf-8"),
            headers={"content-type": "application/json"},
        )
    status, body, headers = spec
    if body is None:
        content = b""
    elif isinstance(body, (bytes, bytearray)):
        content = bytes(body)
    else:
        content = _json.dumps(body).encode()
    return httpx.Response(status_code=status, content=content, headers=headers or {})


def make_sync_transport(
    responses: list[Any], recorder: Recorder
) -> httpx.MockTransport:
    """A sync MockTransport that returns `responses` in order and records each
    incoming request. Reusing the last response if more requests arrive than
    responses were queued keeps the 'exactly N attempts' assertions honest:
    the test asserts recorder.count, not that the queue ran dry."""
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        recorder.record(request)
        spec = queue.pop(0) if len(queue) > 1 else queue[0]
        return _to_response(spec)

    return httpx.MockTransport(handler)


def make_async_transport(
    responses: list[Any], recorder: Recorder
) -> httpx.MockTransport:
    """httpx.MockTransport accepts a sync handler and is usable by both
    httpx.Client and httpx.AsyncClient; the handler body is identical."""
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        recorder.record(request)
        spec = queue.pop(0) if len(queue) > 1 else queue[0]
        return _to_response(spec)

    return httpx.MockTransport(handler)


@pytest.fixture
def recorder() -> Recorder:
    return Recorder()


@pytest.fixture
def make_client(monkeypatch) -> Callable[..., Client]:
    """Build a sync Client wired to a recording mock transport.

    Ensures ASSEMBLYAI_API_KEY is present so construction succeeds; individual
    auth tests override the env explicitly.
    """
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", API_KEY)

    def _build(responses: list[Any], recorder: Recorder, **kwargs) -> Client:
        transport = make_sync_transport(responses, recorder)
        kwargs.setdefault("api_key", API_KEY)
        kwargs.setdefault("base_url", "https://agents.test.local")
        return Client(transport=transport, **kwargs)

    return _build


@pytest.fixture
def make_async_client(monkeypatch) -> Callable[..., AsyncClient]:
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", API_KEY)

    def _build(responses: list[Any], recorder: Recorder, **kwargs) -> AsyncClient:
        transport = make_async_transport(responses, recorder)
        kwargs.setdefault("api_key", API_KEY)
        kwargs.setdefault("base_url", "https://agents.test.local")
        return AsyncClient(transport=transport, **kwargs)

    return _build


@dataclass
class SleepRecorder:
    """Records the upper-bound the SDK derived for each backoff sleep.

    The SDK computes a per-attempt jitter bound `upper = min(cap, base*2**n)`
    and sleeps `random.uniform(0, upper)` (possibly raised to a Retry-After
    lower bound). To make retry tests instant AND assertable, this fixture:

      * pins random.uniform so the actual sleep value == its upper bound, and
      * patches the SDK's sleep functions to record the requested duration
        instead of sleeping.

    `bounds` is the sequence of durations the SDK asked to sleep for, in order.
    """

    bounds: list[float] = field(default_factory=list)


@pytest.fixture
def sleep_recorder(monkeypatch) -> SleepRecorder:
    rec = SleepRecorder()

    # Make jitter deterministic: random.uniform(0, upper) -> upper, so a
    # recorded sleep duration equals the SDK's computed upper bound. The SDK's
    # _retry module references random at module scope (assumption R3).
    import assemblyai_agents._retry as _retry_mod

    monkeypatch.setattr(_retry_mod.random, "uniform", lambda _lo, hi: hi)

    def _record_sync(seconds: float) -> None:
        rec.bounds.append(seconds)

    async def _record_async(seconds: float) -> None:
        rec.bounds.append(seconds)

    # The transport core sleeps via time.sleep (sync) / asyncio.sleep (async),
    # both referenced at the module scope of _retry (assumption R3). Patching
    # the names on that module intercepts every backoff sleep without touching
    # wall-clock time anywhere else.
    monkeypatch.setattr(_retry_mod.time, "sleep", _record_sync)
    monkeypatch.setattr(_retry_mod.asyncio, "sleep", _record_async)

    return rec


def err(code: str, *, request_id: str = "req-test", **extra) -> dict:
    """Build the standard API error envelope used by the mock."""
    body = {
        "code": code,
        "message": f"{code} message",
        "param": None,
        "request_id": request_id,
    }
    body.update(extra)
    return body
