import base64
import json

import pytest
from assemblyai_agents import (
    AsyncClient,
    Client,
    RealtimeError,
)
from assemblyai_agents.models.ws import (
    Code,
    InputSpeechStarted,
    InputSpeechStopped,
    ReplyAudio,
    ReplyDone,
    ReplyStarted,
    SessionEnded,
    SessionError,
    SessionReady,
    SessionUpdatedEvent,
    ToolCall,
    TranscriptAgent,
    TranscriptAgentDelta,
    TranscriptUser,
)
from assemblyai_agents.realtime import (
    UnknownEvent,
    _ws_url_from_base,
)
from websockets.exceptions import (
    ConnectionClosedError,
    ConnectionClosedOK,
    InvalidHandshake,
)
from websockets.frames import Close

API_KEY = "k"


class FakeConnection:
    """In-memory stand-in for a websockets ClientConnection.

    recv() replays the scripted inbound frames in order, then raises the
    scripted close exception (a real websockets.exceptions.ConnectionClosed
    carrying rcvd.code) so the session's __anext__ exercises the exact
    close-code-extraction path it uses in production.
    """

    def __init__(self, inbound=None, close_exc=None):
        # inbound: list of str frames (already json.dumps'd) to replay.
        self._inbound = list(inbound or [])
        # close_exc: the exception recv() raises once inbound is exhausted.
        # Defaults to a clean 1000 close.
        self._close_exc = close_exc or ConnectionClosedOK(
            Close(1000, ""), Close(1000, ""), True
        )
        self.sent: list[str] = []
        self.close_calls = 0

    async def send(self, data):
        self.sent.append(data)

    async def recv(self):
        if self._inbound:
            return self._inbound.pop(0)
        raise self._close_exc

    async def close(self):
        self.close_calls += 1

    @property
    def last_json(self) -> dict:
        return json.loads(self.sent[-1])


def _frame(payload: dict) -> str:
    return json.dumps(payload)


def _install_fake(monkeypatch, fake: FakeConnection) -> dict:
    """Patch realtime.connect so connect() returns `fake` without a real socket.

    Returns a dict the test can inspect for the kwargs connect() was called
    with (notably additional_headers).
    """
    captured: dict = {}

    async def fake_connect(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return fake

    monkeypatch.setattr("assemblyai_agents.realtime.connect", fake_connect)
    return captured


async def _open_session(monkeypatch, fake: FakeConnection, **connect_kwargs):
    captured = _install_fake(monkeypatch, fake)
    client = AsyncClient(api_key=API_KEY)
    session = await client.sessions.connect(**connect_kwargs)
    return session, captured


# --- (a) URL derivation -----------------------------------------------------


def test_ws_url_derivation():
    assert (
        _ws_url_from_base("https://agents.assemblyai.com")
        == "wss://agents.assemblyai.com/v1/ws"
    )
    assert _ws_url_from_base("http://localhost:8080") == "ws://localhost:8080/v1/ws"


# --- (a) bearer header + no token in repr --------------------------


@pytest.mark.asyncio
async def test_connect_sets_bearer_header(monkeypatch):
    fake = FakeConnection()
    session, captured = await _open_session(monkeypatch, fake)

    assert captured["additional_headers"] == {"Authorization": f"Bearer {API_KEY}"}
    # The api_key/token must never appear in the session's repr.
    assert API_KEY not in repr(session)
    assert "Bearer" not in repr(session)


# --- (b) token override beats api_key ---------------------------------------


@pytest.mark.asyncio
async def test_token_override_beats_api_key(monkeypatch):
    fake = FakeConnection()
    _, captured = await _open_session(monkeypatch, fake, token="temp-tok")

    assert captured["additional_headers"] == {"Authorization": "Bearer temp-tok"}


# --- (c) each send method emits the correct wire frame ----------------------


@pytest.mark.asyncio
async def test_send_methods_emit_frames(monkeypatch):
    fake = FakeConnection()
    session, _ = await _open_session(monkeypatch, fake)

    # update: only the non-None allowed keys appear under "session"; never an
    # unlisted key.
    await session.update(system_prompt="be nice", greeting="hi")
    assert fake.last_json == {
        "type": "session.update",
        "session": {"system_prompt": "be nice", "greeting": "hi"},
    }
    sent_session_keys = set(fake.last_json["session"].keys())
    assert sent_session_keys <= {
        "agent_id",
        "system_prompt",
        "greeting",
        "input",
        "output",
        "tools",
        "webhook",
    }
    assert "agent_id" not in sent_session_keys  # None kwargs are omitted

    # update with a nested dict passthrough (input/output) — verbatim.
    await session.update(input={"type": "audio", "format": {"encoding": "audio/pcm"}})
    assert fake.last_json == {
        "type": "session.update",
        "session": {"input": {"type": "audio", "format": {"encoding": "audio/pcm"}}},
    }

    await session.resume("sess-42")
    assert fake.last_json == {"type": "session.resume", "session_id": "sess-42"}

    await session.end()
    assert fake.last_json == {"type": "session.end"}

    await session.cancel_reply("reply-7")
    assert fake.last_json == {"type": "reply.cancel", "reply_id": "reply-7"}

    # create_reply omits "instructions" when None ...
    await session.create_reply()
    assert fake.last_json == {"type": "reply.create"}
    # ... and includes it when given.
    await session.create_reply(instructions="speak slowly")
    assert fake.last_json == {
        "type": "reply.create",
        "instructions": "speak slowly",
    }

    # send_tool_result defaults is_error to False.
    await session.send_tool_result("call-1", "42")
    assert fake.last_json == {
        "type": "tool.result",
        "call_id": "call-1",
        "result": "42",
        "is_error": False,
    }
    await session.send_tool_result("call-2", "boom", is_error=True)
    assert fake.last_json == {
        "type": "tool.result",
        "call_id": "call-2",
        "result": "boom",
        "is_error": True,
    }

    # send_message defaults role to "user".
    await session.send_message("hello")
    assert fake.last_json == {
        "type": "conversation.message",
        "role": "user",
        "content": "hello",
    }
    await session.send_message("sys note", role="system")
    assert fake.last_json == {
        "type": "conversation.message",
        "role": "system",
        "content": "sys note",
    }


# --- (c) send_audio is base64-encoded --------------------------------------


@pytest.mark.asyncio
async def test_send_audio_base64(monkeypatch):
    fake = FakeConnection()
    session, _ = await _open_session(monkeypatch, fake)

    raw = b"\x00\x01\x02"
    await session.send_audio(raw)
    assert fake.last_json == {
        "type": "input.audio",
        "audio": base64.b64encode(raw).decode(),
    }


# --- (d) dispatcher maps every known server type to its model --------------


@pytest.mark.asyncio
async def test_dispatch_known_server_types(monkeypatch):
    # One full, schema-valid frame per the 13 server wire types. Each required
    # field is populated so model_validate succeeds (a thin frame would fall
    # through to UnknownEvent and silently break this assertion).
    frames = [
        ({"type": "input.speech.started", "timestamp": 1.0}, InputSpeechStarted),
        ({"type": "input.speech.stopped", "timestamp": 2.0}, InputSpeechStopped),
        (
            {"type": "reply.audio", "reply_id": "r1", "data": "AAAA"},
            ReplyAudio,
        ),
        (
            {"type": "reply.done", "reply_id": "r1", "status": "completed"},
            ReplyDone,
        ),
        (
            {"type": "reply.started", "reply_id": "r1", "item_id": "i1"},
            ReplyStarted,
        ),
        (
            {"type": "session.ended", "session_duration_seconds": 3.5},
            SessionEnded,
        ),
        (
            {
                "type": "session.error",
                "code": "invalid_audio",
                "message": "bad",
            },
            SessionError,
        ),
        (
            {
                "type": "session.ready",
                "session_id": "sess-1",
                "expires_at": 9999,
                "config": {},
            },
            SessionReady,
        ),
        (
            {"type": "session.updated", "config": {}},
            SessionUpdatedEvent,
        ),
        (
            {
                "type": "tool.call",
                "call_id": "c1",
                "name": "lookup",
                "arguments": {"q": "x"},
            },
            ToolCall,
        ),
        (
            {
                "type": "transcript.agent",
                "item_id": "i1",
                "reply_id": "r1",
                "text": "hi there",
                "interrupted": False,
            },
            TranscriptAgent,
        ),
        (
            {
                "type": "transcript.agent.delta",
                "item_id": "i1",
                "reply_id": "r1",
                "delta": "hel",
                "start_ms": 0,
                "end_ms": 100,
            },
            TranscriptAgentDelta,
        ),
        (
            {"type": "transcript.user", "item_id": "i1", "text": "hello"},
            TranscriptUser,
        ),
    ]

    fake = FakeConnection(inbound=[_frame(p) for p, _ in frames])
    session, _ = await _open_session(monkeypatch, fake)

    received = [event async for event in session]
    assert len(received) == len(frames)
    for event, (_, expected_cls) in zip(received, frames):
        assert isinstance(event, expected_cls), (event, expected_cls)

    # Spot-check that a key field parses on a representative few.
    by_cls = {type(e): e for e in received}
    assert by_cls[SessionReady].session_id == "sess-1"
    assert by_cls[SessionReady].expires_at == 9999
    assert by_cls[SessionError].code == Code.invalid_audio
    assert by_cls[ToolCall].call_id == "c1"
    assert by_cls[TranscriptAgentDelta].delta == "hel"
    assert by_cls[TranscriptUser].text == "hello"


# --- (e) unknown server type -> UnknownEvent, iterator continues ------------


@pytest.mark.asyncio
async def test_unknown_event_fallback(monkeypatch):
    unknown = {"type": "future.event", "x": 1}
    follow = {"type": "transcript.user", "item_id": "i1", "text": "after"}
    fake = FakeConnection(inbound=[_frame(unknown), _frame(follow)])
    session, _ = await _open_session(monkeypatch, fake)

    received = [event async for event in session]
    assert len(received) == 2
    assert isinstance(received[0], UnknownEvent)
    assert received[0].raw == unknown
    # The iterator survived the unknown frame and yielded the next typed event.
    assert isinstance(received[1], TranscriptUser)
    assert received[1].text == "after"


# --- (f) session.error is yielded, not raised ------------------------------


@pytest.mark.asyncio
async def test_session_error_yielded_not_raised(monkeypatch):
    err = {
        "type": "session.error",
        "code": "audio_rate_violation",
        "message": "slow down",
    }
    fake = FakeConnection(inbound=[_frame(err)])
    session, _ = await _open_session(monkeypatch, fake)

    # Must not raise out of the loop.
    received = [event async for event in session]
    assert len(received) == 1
    assert isinstance(received[0], SessionError)
    assert received[0].code == Code.audio_rate_violation


# --- (g) close-code 1008 surfaced on auth-reject ---------------------------


@pytest.mark.asyncio
async def test_close_code_1008_surfaced(monkeypatch):
    err = {
        "type": "session.error",
        "code": "unauthorized",
        "message": "Authentication failed",
    }
    close_exc = ConnectionClosedError(Close(1008, "policy"), None)
    fake = FakeConnection(inbound=[_frame(err)], close_exc=close_exc)
    session, _ = await _open_session(monkeypatch, fake)

    received = []
    # No exception must escape the async-for, despite the abnormal close.
    async for event in session:
        received.append(event)

    assert len(received) == 1
    assert isinstance(received[0], SessionError)
    assert received[0].code == Code.unauthorized
    assert session.close_code == 1008


# --- (h) clean termination on transport close ------------------------------


@pytest.mark.asyncio
async def test_iterator_terminates_on_clean_close(monkeypatch):
    close_exc = ConnectionClosedOK(Close(1000, ""), Close(1000, ""), True)
    fake = FakeConnection(inbound=[], close_exc=close_exc)
    session, _ = await _open_session(monkeypatch, fake)

    received = [event async for event in session]
    assert received == []
    assert session.close_code == 1000


# --- (i) context manager closes socket, idempotent close -------------------


@pytest.mark.asyncio
async def test_context_manager_closes_socket(monkeypatch):
    fake = FakeConnection()
    _install_fake(monkeypatch, fake)
    client = AsyncClient(api_key=API_KEY)

    async with await client.sessions.connect() as s:
        pass

    assert fake.close_calls == 1
    # A second explicit close is a no-op (idempotent) — no extra close call.
    await s.close()
    assert fake.close_calls == 1


# --- handshake failure -> RealtimeError ---------------------------


@pytest.mark.asyncio
async def test_connect_raises_realtimeerror_on_handshake_failure(monkeypatch):
    async def boom(url, **kwargs):
        # A non-101 upgrade surfaces as an InvalidHandshake from websockets.
        raise InvalidHandshake("server rejected WebSocket connection: HTTP 502")

    monkeypatch.setattr("assemblyai_agents.realtime.connect", boom)
    client = AsyncClient(api_key=API_KEY)

    with pytest.raises(RealtimeError):
        await client.sessions.connect()


# --- (j) sync sessions has no connect; async does --------------------------


def test_sync_sessions_has_no_connect():
    assert hasattr(Client(api_key=API_KEY).sessions, "connect") is False
    assert hasattr(AsyncClient(api_key=API_KEY).sessions, "connect") is True


ABNORMAL_DROP_CODE = 1006


def _abnormal_close(code: int = ABNORMAL_DROP_CODE):
    return ConnectionClosedError(Close(code, "dropped"), None)


def _install_fake_queue(monkeypatch, fakes: list) -> list:
    """Patch realtime.connect to pop the NEXT FakeConnection per call and record
    the per-connect kwargs (notably additional_headers).

    Returns a list of the captured-kwargs dicts, one appended per connect() call,
    in call order (index 0 = initial socket, index 1 = first reconnect, ...).
    """
    queue = list(fakes)
    connects: list = []

    async def fake_connect(url, **kwargs):
        record = {"url": url}
        record.update(kwargs)
        connects.append(record)
        if not queue:
            raise AssertionError("connect() called more times than fakes provided")
        return queue.pop(0)

    monkeypatch.setattr("assemblyai_agents.realtime.connect", fake_connect)
    return connects


def _ready_frame(session_id: str, resume_token: str) -> str:
    return _frame(
        {
            "type": "session.ready",
            "session_id": session_id,
            "expires_at": 9999,
            "resume_token": resume_token,
            "config": {},
        }
    )


def _bearer(connect_record: dict) -> str:
    return connect_record["additional_headers"]["Authorization"]


@pytest.mark.asyncio
async def test_auto_resume_off_terminates_on_drop(monkeypatch):
    ready = _ready_frame("sess-1", "rt-1")
    fake = FakeConnection(inbound=[ready], close_exc=_abnormal_close(1006))
    connects = _install_fake_queue(monkeypatch, [fake])
    client = AsyncClient(api_key=API_KEY)
    session = await client.sessions.connect()

    received = [event async for event in session]

    assert len(received) == 1
    assert isinstance(received[0], SessionReady)
    assert session.close_code == 1006
    assert len(connects) == 1


@pytest.mark.asyncio
async def test_auto_resume_reconnects_with_resume_token_bearer(monkeypatch):
    fake1 = FakeConnection(
        inbound=[_ready_frame("sess-1", "rt-1")],
        close_exc=_abnormal_close(1006),
    )
    fake2 = FakeConnection(inbound=[_ready_frame("sess-1", "rt-2")])
    connects = _install_fake_queue(monkeypatch, [fake1, fake2])
    client = AsyncClient(api_key=API_KEY)
    session = await client.sessions.connect(auto_resume=True)

    received = [event async for event in session]

    assert [type(e) for e in received] == [SessionReady, SessionReady]
    assert len(connects) == 2
    assert _bearer(connects[0]) == f"Bearer {API_KEY}"
    assert _bearer(connects[1]) == "Bearer rt-1"
    assert API_KEY not in _bearer(connects[1])
    assert json.loads(fake2.sent[0]) == {
        "type": "session.resume",
        "session_id": "sess-1",
    }


@pytest.mark.asyncio
async def test_auto_resume_continues_iteration_after_reconnect(monkeypatch):
    fake1 = FakeConnection(
        inbound=[_ready_frame("sess-1", "rt-1")],
        close_exc=_abnormal_close(1006),
    )
    fake2 = FakeConnection(
        inbound=[
            _ready_frame("sess-1", "rt-2"),
            _frame({"type": "transcript.user", "item_id": "i1", "text": "after"}),
        ]
    )
    _install_fake_queue(monkeypatch, [fake1, fake2])
    client = AsyncClient(api_key=API_KEY)
    session = await client.sessions.connect(auto_resume=True)

    received = [event async for event in session]

    assert [type(e) for e in received] == [
        SessionReady,
        SessionReady,
        TranscriptUser,
    ]
    assert received[-1].text == "after"
    assert [json.loads(s)["type"] for s in fake2.sent] == ["session.resume"]


@pytest.mark.asyncio
async def test_auto_resume_gives_up_after_max_attempts(monkeypatch):
    max_attempts = 3
    fake1 = FakeConnection(
        inbound=[_ready_frame("sess-1", "rt-1")],
        close_exc=_abnormal_close(1006),
    )
    connects: list = []
    queue = [fake1]

    async def fake_connect(url, **kwargs):
        connects.append({"url": url, **kwargs})
        if queue:
            return queue.pop(0)
        raise InvalidHandshake("server rejected WebSocket connection: HTTP 502")

    monkeypatch.setattr("assemblyai_agents.realtime.connect", fake_connect)
    client = AsyncClient(api_key=API_KEY)
    session = await client.sessions.connect(
        auto_resume=True, max_resume_attempts=max_attempts
    )

    with pytest.raises(RealtimeError) as excinfo:
        async for _ in session:
            pass

    assert excinfo.value.close_code == 1006
    assert len(connects) == 1 + max_attempts


@pytest.mark.asyncio
async def test_auto_resume_terminal_on_session_not_found(monkeypatch):
    fake1 = FakeConnection(
        inbound=[_ready_frame("sess-1", "rt-1")],
        close_exc=_abnormal_close(1006),
    )
    fake2 = FakeConnection(
        inbound=[
            _frame(
                {
                    "type": "session.error",
                    "code": "session_not_found",
                    "message": "no such session",
                }
            )
        ]
    )
    connects = _install_fake_queue(monkeypatch, [fake1, fake2])
    client = AsyncClient(api_key=API_KEY)
    session = await client.sessions.connect(auto_resume=True, max_resume_attempts=5)

    with pytest.raises(RealtimeError):
        async for _ in session:
            pass

    assert len(connects) == 2


@pytest.mark.asyncio
async def test_resume_token_recaptured_each_session_ready(monkeypatch):
    fake1 = FakeConnection(
        inbound=[_ready_frame("sess-1", "rt-1")],
        close_exc=_abnormal_close(1006),
    )
    fake2 = FakeConnection(
        inbound=[_ready_frame("sess-1", "rt-2")],
        close_exc=_abnormal_close(1006),
    )
    fake3 = FakeConnection(inbound=[_ready_frame("sess-1", "rt-3")])
    connects = _install_fake_queue(monkeypatch, [fake1, fake2, fake3])
    client = AsyncClient(api_key=API_KEY)
    session = await client.sessions.connect(auto_resume=True)

    received = [event async for event in session]

    assert [type(e) for e in received] == [SessionReady, SessionReady, SessionReady]
    assert len(connects) == 3
    assert _bearer(connects[1]) == "Bearer rt-1"
    assert _bearer(connects[2]) == "Bearer rt-2"
    assert json.loads(fake2.sent[0]) == {
        "type": "session.resume",
        "session_id": "sess-1",
    }
    assert json.loads(fake3.sent[0]) == {
        "type": "session.resume",
        "session_id": "sess-1",
    }


@pytest.mark.asyncio
async def test_auto_resume_empty_token_terminates_on_drop(monkeypatch):
    ready = _ready_frame("sess-1", "")
    fake = FakeConnection(inbound=[ready], close_exc=_abnormal_close(1006))
    connects = _install_fake_queue(monkeypatch, [fake])
    client = AsyncClient(api_key=API_KEY)
    session = await client.sessions.connect(auto_resume=True)

    received = [event async for event in session]

    assert len(received) == 1
    assert isinstance(received[0], SessionReady)
    assert session.close_code == 1006
    assert len(connects) == 1


@pytest.mark.asyncio
async def test_auto_resume_clean_close_terminates(monkeypatch):
    fake = FakeConnection(
        inbound=[_ready_frame("sess-1", "rt-1")],
        close_exc=ConnectionClosedOK(Close(1000, ""), Close(1000, ""), True),
    )
    connects = _install_fake_queue(monkeypatch, [fake])
    client = AsyncClient(api_key=API_KEY)
    session = await client.sessions.connect(auto_resume=True)

    received = [event async for event in session]

    assert [type(e) for e in received] == [SessionReady]
    assert session.close_code == 1000
    assert len(connects) == 1


@pytest.mark.asyncio
async def test_auto_resume_send_side_drop_retries(monkeypatch):
    fake1 = FakeConnection(
        inbound=[_ready_frame("sess-1", "rt-1")],
        close_exc=_abnormal_close(1006),
    )
    fake_send_drop = FakeConnection(inbound=[_ready_frame("sess-1", "rt-2")])

    async def _raise_on_send(_data):
        raise _abnormal_close(1006)

    fake_send_drop.send = _raise_on_send
    fake2 = FakeConnection(inbound=[_ready_frame("sess-1", "rt-2")])
    connects = _install_fake_queue(monkeypatch, [fake1, fake_send_drop, fake2])
    client = AsyncClient(api_key=API_KEY)
    session = await client.sessions.connect(auto_resume=True, max_resume_attempts=5)

    received = [event async for event in session]

    assert [type(e) for e in received] == [SessionReady, SessionReady]
    assert len(connects) == 3
    assert fake_send_drop.close_calls == 1
    assert _bearer(connects[2]) == "Bearer rt-1"


@pytest.mark.asyncio
async def test_auto_resume_connect_error_retries(monkeypatch):
    fake1 = FakeConnection(
        inbound=[_ready_frame("sess-1", "rt-1")],
        close_exc=_abnormal_close(1006),
    )
    good = FakeConnection(inbound=[_ready_frame("sess-1", "rt-2")])
    script = [fake1, ConnectionRefusedError("refused"), good]
    connects: list = []

    async def fake_connect(url, **kwargs):
        connects.append({"url": url, **kwargs})
        item = script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr("assemblyai_agents.realtime.connect", fake_connect)
    client = AsyncClient(api_key=API_KEY)
    session = await client.sessions.connect(auto_resume=True, max_resume_attempts=5)

    received = [event async for event in session]

    assert [type(e) for e in received] == [SessionReady, SessionReady]
    assert len(connects) == 3
    assert _bearer(connects[2]) == "Bearer rt-1"
