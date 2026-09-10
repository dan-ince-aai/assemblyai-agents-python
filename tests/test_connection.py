import asyncio
import base64

import pytest
from assemblyai_agents import (
    AssemblyAIAgentsError,
    DeviceAudioNotInstalledError,
    audio_io,
)
from assemblyai_agents import connection as connection_mod

# DeviceAudioError is NOT re-exported from the package barrel, so it can only be
# reached through the module; AssemblyAIAgentsError (its base) is the exported
# name a caller can actually catch it by.
from assemblyai_agents.audio_io import DeviceAudioError
from assemblyai_agents.connection import AgentConnection, ToolRouter
from assemblyai_agents.models.ws import (
    InputSpeechStarted,
    ReplyAudio,
    ReplyDone,
    ReplyStarted,
    SessionEnded,
    ToolCall,
    TranscriptAgent,
    TranscriptUser,
)
from websockets.exceptions import ConnectionClosedError
from websockets.frames import Close

pytestmark = pytest.mark.asyncio

MINTED_TOKEN = "tok-minted-short-lived"
API_KEY = "k-raw-api-key-must-not-leak"
AGENT_ID = "ag-123"


# --------------------------------------------------------------------------- #
# Faithful fakes modeling the real surface.                          #
# --------------------------------------------------------------------------- #
def _reply_audio(reply_id: str, pcm: bytes) -> ReplyAudio:
    # ReplyAudio.data is base64 (ws.py:105-109); base64_to_pcm validates it
    # (audio.py:21-26), so the payload must be genuine base64.
    return ReplyAudio(reply_id=reply_id, data=base64.b64encode(pcm).decode())


class FakeTokenResponse:
    def __init__(self, token: str) -> None:
        self.token = token
        self.expires_at = None


class FakeAgentResponse:
    def __init__(self, agent_id: str) -> None:
        self.id = agent_id
        self.name = "fake-agent"


class FakeTokens:
    def __init__(self) -> None:
        self.create_calls = 0

    async def create(self, body=None):
        self.create_calls += 1
        return FakeTokenResponse(MINTED_TOKEN)


class FakeAgents:
    def __init__(self) -> None:
        self.created = []
        self.gets = []

    async def create(self, req):
        self.created.append(req)
        return FakeAgentResponse("ag-created-999")

    async def get(self, agent_id):
        self.gets.append(agent_id)
        return FakeAgentResponse(agent_id)


class FakeSession:
    """Async-iterable stand-in for AsyncRealtimeSession.

    Yields a SCRIPTED list of events in order, then stops. Records every
    send/update/lifecycle call so a test can assert what AgentConnection drove.

    Single-reader instrumentation: __anext__ asserts only ONE iterator is ever
    created and the events are pulled by exactly one consumer; a 2nd __aiter__
    (a second reader) trips the invariant. recv() raises if touched — the mic
    pump must never read the socket (solution.md:13; PR15b invariant).
    """

    def __init__(self, events=None, *, stop="stop"):
        self._events = list(events or [])
        # stop="stop" -> raise StopAsyncIteration after the script;
        # stop="ended" -> yield a SessionEnded as the final event instead.
        if stop == "ended":
            self._events.append(SessionEnded(session_duration_seconds=1.0))
        self._i = 0
        self._iter_handed_out = 0
        self.updates = []
        self.tool_results = []
        self.messages = []
        self.audio_sent = []
        self.end_calls = 0
        self.close_calls = 0

    def __aiter__(self):
        self._iter_handed_out += 1
        assert self._iter_handed_out == 1, (
            "single-reader invariant: AgentConnection must own the ONLY async-for "
            "over the session (a 2nd iterator means a 2nd reader)"
        )
        return self

    async def __anext__(self):
        if self._i >= len(self._events):
            raise StopAsyncIteration
        event = self._events[self._i]
        self._i += 1
        # Yield control so any concurrent (illegal) reader would interleave and
        # the single-reader assertions above would fire.
        await asyncio.sleep(0)
        return event

    async def recv(self):
        raise AssertionError("AgentConnection must iterate, never call recv directly")

    async def update(self, **kwargs):
        self.updates.append(kwargs)

    async def send_tool_result(self, call_id, result, *, is_error=False):
        self.tool_results.append(
            {"call_id": call_id, "result": result, "is_error": is_error}
        )

    async def send_message(self, content, *, role="user"):
        self.messages.append({"content": content, "role": role})

    async def send_audio(self, audio):
        self.audio_sent.append(bytes(audio))

    async def end(self):
        self.end_calls += 1

    async def close(self):
        self.close_calls += 1

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.close()


class FakeSessions:
    """Records the kwargs sessions.connect was called with so a test can assert
    the bearer is the minted token (NOT the raw api_key)."""

    def __init__(self, session: FakeSession) -> None:
        self._session = session
        self.connect_kwargs = None
        self.connect_calls = 0

    async def connect(self, *, url=None, token=None, auto_resume=False, **kwargs):
        self.connect_calls += 1
        self.connect_kwargs = {
            "url": url,
            "token": token,
            "auto_resume": auto_resume,
            **kwargs,
        }
        return self._session


class FakeClient:
    """Stand-in for AsyncClient: exposes .tokens / .sessions / .agents and an
    api_key the test asserts is NEVER used as the connect bearer."""

    def __init__(self, session: FakeSession) -> None:
        self.api_key = API_KEY
        self.tokens = FakeTokens()
        self.sessions = FakeSessions(session)
        self.agents = FakeAgents()


class FakeSink:
    """Records every event forwarded to PlaybackSink.handle, plus open/aclose().

    Faithful to the surface AgentConnection.run composes: the real PlaybackSink
    starts PortAudio in __aenter__ -> _open() (audio_io.py:133-155) and stops it
    in aclose(), and handle(event) only touches the deque, so a sink that is
    never entered buffers decoded PCM that nothing ever plays. `log` records the
    ORDER of those calls so a test can prove the device was opened before the
    first frame was written.
    """

    def __init__(self) -> None:
        self.handled = []
        self.aclose_calls = 0
        self.aenter_calls = 0
        self.log = []

    async def __aenter__(self) -> "FakeSink":
        self.aenter_calls += 1
        self.log.append("open")
        return self

    async def __aexit__(self, *exc_info) -> None:
        await self.aclose()

    def handle(self, event) -> None:
        self.handled.append(event)
        self.log.append("handle")

    async def aclose(self) -> None:
        self.aclose_calls += 1
        self.log.append("close")


async def _healthy_mic(session) -> None:
    """Stand-in for a working microphone_stream: parks forever like the real
    pump's `while True` capture loop, and is only ever ended by cancellation."""
    await asyncio.Event().wait()


def _quiet_mic(monkeypatch) -> None:
    # The mic pump is started whenever a sink exists. These tests are about
    # playback/dispatch, not capture, and pyaudio is NOT a dep of this target —
    # so stand in a healthy pump rather than letting the real one fail on import.
    monkeypatch.setattr(connection_mod, "microphone_stream", _healthy_mic)


def _make_agent(events, *, audio=True, sink=None, tools=None, **kwargs):
    """Build an AgentConnection over a FakeClient/FakeSession.

    When audio=True and a `sink` is given, inject it so run() forwards events to
    it WITHOUT opening a real device (no pyaudio). The injection point is the
    private `_sink` attribute the impl uses internally; if the impl exposes a
    different seam, adjust here (single place).
    """
    session = FakeSession(events)
    client = FakeClient(session)
    agent = AgentConnection(
        agent_id=AGENT_ID,
        client=client,
        audio=audio,
        tools=tools,
        **kwargs,
    )
    return agent, client, session, sink


# --------------------------------------------------------------------------- #
# 1. enter mints a token and connects with it (not the api_key)               #
# --------------------------------------------------------------------------- #
async def test_enter_mints_token_and_connects(monkeypatch):
    session = FakeSession([])
    client = FakeClient(session)

    # (a) default: __aenter__ mints a short-lived token and uses it as the bearer.
    agent = AgentConnection(agent_id=AGENT_ID, client=client, audio=False)
    async with agent:
        pass

    assert client.tokens.create_calls == 1, "must mint a session token"
    assert client.sessions.connect_calls == 1
    assert client.sessions.connect_kwargs["token"] == MINTED_TOKEN, (
        "the minted token must be the connect bearer"
    )
    assert client.sessions.connect_kwargs["token"] != API_KEY, (
        "the raw api_key must NEVER be passed as the ws bearer"
    )

    # (b) caller-supplied token=: skips the mint, passes the token straight through.
    session2 = FakeSession([])
    client2 = FakeClient(session2)
    agent2 = AgentConnection(
        agent_id=AGENT_ID, client=client2, audio=False, token="caller-token"
    )
    async with agent2:
        pass

    assert client2.tokens.create_calls == 0, "token= must skip the mint"
    assert client2.sessions.connect_kwargs["token"] == "caller-token"


# --------------------------------------------------------------------------- #
# 2. enter binds the agent id (and nothing else)                              #
# --------------------------------------------------------------------------- #
async def test_enter_binds_agent_id():
    session = FakeSession([])
    client = FakeClient(session)
    agent = AgentConnection(agent_id=AGENT_ID, client=client, audio=False)

    async with agent:
        pass

    assert session.updates, "must call session.update to bind the agent"
    bind = session.updates[0]
    assert bind.get("agent_id") == AGENT_ID
    # No inline config: only agent_id is bound (no system_prompt/greeting).
    assert "system_prompt" not in bind or bind["system_prompt"] is None
    assert "greeting" not in bind or bind["greeting"] is None


# --------------------------------------------------------------------------- #
# 3. run forwards reply audio/started to the sink's handle                    #
# --------------------------------------------------------------------------- #
async def test_run_routes_reply_audio_to_sink_handle(monkeypatch):
    _quiet_mic(monkeypatch)
    started = ReplyStarted(item_id="it-1", reply_id="r1")
    audio_evt = _reply_audio("r1", b"\x01\x02" * 240)
    events = [started, audio_evt, SessionEnded(session_duration_seconds=1.0)]

    sink = FakeSink()
    agent, client, session, _ = _make_agent(events, audio=True, sink=sink)
    agent._sink = sink  # inject fake sink (no real device)

    async with agent:
        await agent.run()

    assert started in sink.handled, "ReplyStarted must reach sink.handle"
    assert audio_evt in sink.handled, "ReplyAudio must reach sink.handle"


# --------------------------------------------------------------------------- #
# 4. barge-in: InputSpeechStarted flushes via the sink, stream not closed     #
# --------------------------------------------------------------------------- #
async def test_run_flushes_on_barge_in(monkeypatch):
    _quiet_mic(monkeypatch)
    barge = InputSpeechStarted()
    events = [
        _reply_audio("r1", b"\x05\x06" * 240),
        barge,
        SessionEnded(session_duration_seconds=1.0),
    ]
    sink = FakeSink()
    agent, client, session, _ = _make_agent(events, audio=True, sink=sink)
    agent._sink = sink

    async with agent:
        await agent.run()

    assert barge in sink.handled, (
        "InputSpeechStarted must reach sink.handle (barge-in is delegated there)"
    )
    # The sink is aclosed exactly once on teardown — never mid-run, never twice.
    assert sink.aclose_calls == 1


# --------------------------------------------------------------------------- #
# 5. tool call → handler(**args) → send_tool_result; raise/unknown → is_error #
# --------------------------------------------------------------------------- #
async def test_run_routes_tool_call_and_replies():
    seen = {}

    async def get_order(a):
        seen["a"] = a
        return "order-OK"

    def boom():
        raise RuntimeError("handler exploded")

    events = [
        ToolCall(name="get_order", call_id="c1", arguments={"a": 1}),
        ToolCall(name="boom", call_id="c2", arguments={}),
        ToolCall(name="nope", call_id="c3", arguments={}),
        SessionEnded(session_duration_seconds=1.0),
    ]
    agent, client, session, _ = _make_agent(
        events, audio=False, tools={"get_order": get_order, "boom": boom}
    )

    async with agent:
        await agent.run()

    by_id = {r["call_id"]: r for r in session.tool_results}

    # registered handler invoked with splatted arguments, result replied (ok).
    assert seen == {"a": 1}, "handler must be called with **arguments"
    assert by_id["c1"]["is_error"] is False
    assert "order-OK" in str(by_id["c1"]["result"])

    # handler that raises -> error tool-result, no exception out of run().
    assert by_id["c2"]["is_error"] is True
    assert str(by_id["c2"]["result"])  # non-empty message

    # unregistered name -> error tool-result, never an unhandled exception.
    assert by_id["c3"]["is_error"] is True
    assert str(by_id["c3"]["result"])


# --------------------------------------------------------------------------- #
# 6. transcripts fan out to sync AND async callbacks                          #
# --------------------------------------------------------------------------- #
async def test_run_fans_out_transcripts():
    user_seen = []
    agent_seen = []
    delta_seen = []

    events = [
        TranscriptUser(item_id="u1", text="hello"),
        TranscriptAgent(
            interrupted=False, item_id="a1", reply_id="r1", text="hi there"
        ),
        SessionEnded(session_duration_seconds=1.0),
    ]
    agent, client, session, _ = _make_agent(events, audio=False)

    # sync callback (assignment).
    def _on_user(text):
        user_seen.append(text)

    # async callback (decorator) — must be awaited.
    @agent.on_agent_transcript
    async def _on_agent(text):
        agent_seen.append(text)

    # a third callback to prove the generic fan-out path (delta), sync.
    def _on_delta(text):
        delta_seen.append(text)

    agent.on_user_transcript(_on_user)
    agent.on_agent_delta(_on_delta)

    async with agent:
        await agent.run()

    assert user_seen == ["hello"], "sync user-transcript callback must fire"
    assert agent_seen == ["hi there"], "async agent-transcript callback must be awaited"


# --------------------------------------------------------------------------- #
# 7. audio=False: no device/sink; ReplyAudio goes to on_agent_audio (BYO)     #
# --------------------------------------------------------------------------- #
async def test_audio_false_uses_byo_callback(monkeypatch):
    # Hard guard: opening a PlaybackSink would import pyaudio. If the impl tried
    # to open a device under audio=False, this stub turns it into a loud failure
    # instead of a silent real-pyaudio import.
    def _no_device(*a, **k):
        raise AssertionError("audio=False must NOT construct/open a device sink")

    monkeypatch.setattr(audio_io, "PlaybackSink", _no_device)

    pcm = b"\x09\x09" * 240
    events = [
        _reply_audio("r1", pcm),
        SessionEnded(session_duration_seconds=1.0),
    ]
    delivered = []

    agent, client, session, _ = _make_agent(events, audio=False)
    agent.on_agent_audio(lambda evt: delivered.append(evt))

    async with agent:
        await agent.run()

    assert agent._sink is None, "audio=False must open no sink"
    assert len(delivered) == 1, "ReplyAudio must be delivered to on_agent_audio"
    assert delivered[0].reply_id == "r1"


# --------------------------------------------------------------------------- #
# 8. single-reader invariant: AgentConnection owns the only async-for         #
# --------------------------------------------------------------------------- #
async def test_single_reader_invariant():
    events = [
        TranscriptUser(item_id="u1", text="one"),
        TranscriptUser(item_id="u2", text="two"),
        SessionEnded(session_duration_seconds=1.0),
    ]
    agent, client, session, _ = _make_agent(events, audio=False)

    # FakeSession.__aiter__ asserts exactly ONE iterator is handed out, and
    # recv() raises if touched. run() must drive the single async-for; the mic
    # pump (not started under audio=False) must never read the socket.
    async with agent:
        await agent.run()

    assert session._iter_handed_out == 1, "exactly one reader (run's async-for)"
    assert session._i == len(session._events), "all events consumed once"


# --------------------------------------------------------------------------- #
# 9. teardown is idempotent: cancel mic, aclose sink, end session once        #
# --------------------------------------------------------------------------- #
async def test_teardown_idempotent():
    sink = FakeSink()
    agent, client, session, _ = _make_agent([], audio=True, sink=sink)
    agent._sink = sink

    await agent.__aenter__()
    await agent.__aexit__(None, None, None)
    # Second exit (and an idle re-teardown) must be a no-op, not an error.
    await agent.__aexit__(None, None, None)

    assert sink.aclose_calls == 1, "sink aclosed exactly once"
    assert session.end_calls <= 1, "session ended at most once (idempotent)"
    # No exception raised by the double teardown == pass.


async def test_teardown_survives_session_end_drop():
    # The server closes the socket right after session.ended, so session.end()
    # can raise ConnectionClosed at teardown — __aexit__ must swallow it so it
    # doesn't mask the real exit reason.
    agent, client, session, _ = _make_agent([], audio=False)
    await agent.__aenter__()

    async def _end_on_dead_socket():
        session.end_calls += 1
        raise ConnectionClosedError(Close(1006, "server closed"), None)

    session.end = _end_on_dead_socket
    # Must NOT raise out of __aexit__.
    await agent.__aexit__(None, None, None)
    assert session.end_calls == 1


# --------------------------------------------------------------------------- #
# 11. PlaybackSink.handle unit: the four event arms (no real stream)          #
# --------------------------------------------------------------------------- #
async def test_playback_sink_handle_routes_events():
    # Constructed but NOT opened: handle() only touches the deque + interrupted
    # set + base64 decode (audio_io.py:154-167), so no PortAudio stream is needed.
    sink = audio_io.PlaybackSink(frames_per_buffer=480)

    pcm = b"\x11\x22" * 240

    # ReplyStarted -> track the current reply id.
    sink.handle(ReplyStarted(item_id="it-1", reply_id="r1"))

    # ReplyAudio -> enqueue decoded PCM onto the buffer (via write()).
    sink.handle(_reply_audio("r1", pcm))
    assert list(sink._buffer) == [pcm], "ReplyAudio must enqueue decoded PCM"

    # InputSpeechStarted -> flush queued audio AND mark the current reply interrupted.
    sink.handle(InputSpeechStarted())
    assert list(sink._buffer) == [], "InputSpeechStarted must flush queued audio"
    assert "r1" in sink._interrupted, (
        "InputSpeechStarted must mark the in-flight reply interrupted"
    )

    # After interruption, further audio for that reply id is dropped by write().
    sink.handle(_reply_audio("r1", pcm))
    assert list(sink._buffer) == [], "audio for an interrupted reply must be dropped"

    # ReplyDone(interrupted) -> backstop mark_interrupted for its reply id.
    sink.handle(ReplyDone(reply_id="r2", status="interrupted"))
    assert "r2" in sink._interrupted, "ReplyDone(interrupted) is the backstop"

    # Unrelated event -> ignored (no raise, no state change).
    before_buf = list(sink._buffer)
    before_int = set(sink._interrupted)
    sink.handle(TranscriptUser(item_id="u9", text="ignored"))
    assert list(sink._buffer) == before_buf
    assert set(sink._interrupted) == before_int


# --------------------------------------------------------------------------- #
# 12. DEFECT 1: run() must OPEN the playback sink, or decoded audio is never  #
#     played — the caller hears silence on an otherwise healthy session.      #
# --------------------------------------------------------------------------- #
class ExplodingSession(FakeSession):
    """Yields its script, then raises instead of stopping cleanly — models the
    run loop dying mid-session (a dropped socket, a bad frame)."""

    def __init__(self, events, exc):
        super().__init__(events)
        self._exc = exc

    async def __anext__(self):
        if self._i >= len(self._events):
            raise self._exc
        return await super().__anext__()


class HangingSession(FakeSession):
    """Yields its script, then parks forever — models a live socket that has no
    further server events, so a test can cancel run() mid-session."""

    async def __anext__(self):
        if self._i >= len(self._events):
            await asyncio.Event().wait()
        return await super().__anext__()


def _agent_with_sink(session, sink, monkeypatch=None):
    client = FakeClient(session)
    agent = AgentConnection(agent_id=AGENT_ID, client=client, audio=True)
    agent._sink = sink
    if monkeypatch is not None:
        _quiet_mic(monkeypatch)
    return agent


async def test_run_opens_sink_before_first_frame_and_closes_after(monkeypatch):
    audio_evt = _reply_audio("r1", b"\x01\x02" * 240)
    events = [
        ReplyStarted(item_id="it-1", reply_id="r1"),
        audio_evt,
        SessionEnded(session_duration_seconds=1.0),
    ]
    sink = FakeSink()
    agent = _agent_with_sink(FakeSession(events), sink, monkeypatch)

    async with agent:
        await agent.run()

    assert sink.aenter_calls == 1, (
        "run() must OPEN the playback sink exactly once — an unopened sink "
        "buffers decoded PCM that no PortAudio stream ever drains, so the "
        "caller hears silence"
    )
    assert sink.log[0] == "open", "the device must be opened BEFORE any frame"
    assert sink.log.index("open") < sink.log.index("handle")
    assert audio_evt in sink.handled, "decoded frames must reach the opened sink"
    assert sink.log[-1] == "close", "the sink must be closed on the way out"
    assert sink.aclose_calls == 1


async def test_run_closes_sink_when_loop_raises(monkeypatch):
    boom = RuntimeError("socket died mid-session")
    sink = FakeSink()
    agent = _agent_with_sink(
        ExplodingSession([_reply_audio("r1", b"\x03\x04" * 240)], boom),
        sink,
        monkeypatch,
    )

    async with agent:
        with pytest.raises(RuntimeError, match="socket died mid-session"):
            await agent.run()

    assert sink.aenter_calls == 1, "the sink was opened"
    assert sink.aclose_calls == 1, (
        "an opened sink MUST be closed on the exception path too — otherwise the "
        "failed session holds the audio device open"
    )


async def test_run_closes_sink_on_cancellation(monkeypatch):
    sink = FakeSink()
    agent = _agent_with_sink(
        HangingSession([_reply_audio("r1", b"\x07\x08" * 240)]), sink, monkeypatch
    )

    async with agent:
        task = asyncio.create_task(agent.run())
        # Let run() open the sink and consume the scripted frame, then park.
        for _ in range(6):
            await asyncio.sleep(0)
        assert sink.aenter_calls == 1, "the sink was opened before we cancelled"
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert sink.aclose_calls == 1, (
        "a cancelled session MUST still release the audio device"
    )


# --------------------------------------------------------------------------- #
# 13. DEFECT 2: a dead microphone must not fail invisibly. The pump is a      #
#     background task; if nobody retrieves its exception the caller talks to  #
#     an agent that cannot hear them, with no error anywhere.                 #
# --------------------------------------------------------------------------- #
async def test_mic_failure_surfaces_out_of_run(monkeypatch):
    async def _dead_mic(session):
        raise DeviceAudioError("failed to open input audio device: [Errno -9996]")

    monkeypatch.setattr(connection_mod, "microphone_stream", _dead_mic)

    sink = FakeSink()
    # The socket parks forever: nothing but the mic failure can end this session,
    # so a swallowed mic error means run() hangs / returns with no signal.
    agent = _agent_with_sink(HangingSession([]), sink)

    async with agent:
        with pytest.raises(DeviceAudioError) as excinfo:
            await asyncio.wait_for(agent.run(), timeout=5)

    msg = str(excinfo.value)
    # What was wrong, why it matters, and the specific next action.
    assert "failed to open input audio device" in msg, "must keep the real cause"
    assert "microphone" in msg.lower()
    assert "audio=False" in msg, "must name the concrete escape hatch"
    assert sink.aclose_calls == 1, "the sink is still released on this path"


async def test_mic_failure_is_catchable_as_sdk_error(monkeypatch):
    # DeviceAudioNotInstalledError derives from ImportError, so on its own it is
    # NOT caught by `except AssemblyAIAgentsError`. Surfacing it as a
    # DeviceAudioError (an AssemblyAIAgentsError) makes the SDK's own base class
    # sufficient for a caller to handle a dead mic.
    async def _no_pyaudio_mic(session):
        raise DeviceAudioNotInstalledError()

    monkeypatch.setattr(connection_mod, "microphone_stream", _no_pyaudio_mic)

    agent = _agent_with_sink(HangingSession([]), FakeSink())

    async with agent:
        with pytest.raises(AssemblyAIAgentsError) as excinfo:
            await asyncio.wait_for(agent.run(), timeout=5)

    assert isinstance(excinfo.value, DeviceAudioError)
    assert "assemblyai-agents[audio]" in str(excinfo.value), (
        "the install instruction from the underlying cause must survive"
    )


async def test_healthy_mic_does_not_disturb_the_run(monkeypatch):
    # The guard against over-correcting: a mic that is simply cancelled at
    # teardown is NOT a failure and must not raise out of run().
    _quiet_mic(monkeypatch)
    sink = FakeSink()
    agent = _agent_with_sink(
        FakeSession([SessionEnded(session_duration_seconds=1.0)]), sink
    )

    async with agent:
        await agent.run()

    assert sink.aclose_calls == 1


# --------------------------------------------------------------------------- #
# ToolRouter unit (backs criterion 5; the single-homed dispatch primitive)    #
# --------------------------------------------------------------------------- #
async def test_tool_router_handle_dispatches_and_errors():
    calls = {}

    async def ok(x):
        calls["x"] = x
        return "done"

    router = ToolRouter({"ok": ok})
    session = FakeSession([])

    await router.handle(ToolCall(name="ok", call_id="c1", arguments={"x": 7}), session)
    assert calls == {"x": 7}
    assert session.tool_results[0]["call_id"] == "c1"
    assert session.tool_results[0]["is_error"] is False

    # add() + unknown-name path, and a non-ToolCall event is ignored.
    await router.handle(ToolCall(name="missing", call_id="c2", arguments={}), session)
    assert session.tool_results[1]["is_error"] is True

    pre = len(session.tool_results)
    await router.handle(TranscriptUser(item_id="u1", text="x"), session)
    assert len(session.tool_results) == pre, "non-ToolCall events are ignored"
