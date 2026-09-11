import asyncio
import inspect
from typing import Any, Callable, Optional

from websockets.exceptions import ConnectionClosed

from ._client import AsyncClient
from ._exceptions import RealtimeError
from .audio_io import DeviceAudioError, PlaybackSink, microphone_stream
from .models.ws import (
    ReplyAudio,
    SessionEnded,
    SessionError,
    SessionReady,
    ToolCall,
    TranscriptAgent,
    TranscriptAgentDelta,
    TranscriptUser,
)
from .realtime import AsyncRealtimeSession


async def _maybe_await(result: Any) -> Any:
    if inspect.isawaitable(result):
        return await result
    return result


class AgentConnection:
    """A live connection to an already-deployed agent, keyed by ``agent_id``.

    A test client: the microphone and speaker end of a call, for talking to an
    agent from a terminal or building your own audio transport. It mints a
    token, opens the WebSocket, binds the deployed agent with
    ``session.update(agent_id=...)`` and owns the run loop. Nothing about the
    agent is configured here, and no tool runs here — tools are served over
    HTTPS by the process that declared them, on a phone call and a WebSocket
    session alike.
    """

    def __init__(
        self,
        *,
        agent_id: str,
        api_key: Optional[str] = None,
        client: Optional[AsyncClient] = None,
        audio: bool = True,
        auto_resume: bool = True,
        url: Optional[str] = None,
        token: Optional[str] = None,
    ) -> None:
        self._agent_id = agent_id
        self._client = client or AsyncClient(api_key=api_key)
        # None when audio=False (BYO transport); the run loop branches on this and
        # the tests inject a fake sink onto this exact attribute.
        self._sink: Optional[PlaybackSink] = PlaybackSink() if audio else None
        self._auto_resume = auto_resume
        self._url = url
        self._token = token
        self._session: Optional[AsyncRealtimeSession] = None
        self._mic: Optional[asyncio.Task] = None
        # First failure out of the mic pump, kept so run() can report it instead
        # of the caller discovering it by being unheard for the whole call.
        self._mic_error: Optional[BaseException] = None
        self._run_task: Optional[asyncio.Task] = None
        self._closed = False
        # The sink is aclosed at most once whether run()'s finally or teardown
        # gets there first (the FakeSink in the tests is not self-idempotent).
        self._sink_closed = False
        self._callbacks: dict[str, list[Callable]] = {
            "ready": [],
            "user_transcript": [],
            "agent_transcript": [],
            "agent_delta": [],
            "agent_audio": [],
            "error": [],
        }

    def _register(self, slot: str, fn: Callable) -> Callable:
        # Return fn so the registrar works both as @decorator and as a direct call.
        self._callbacks[slot].append(fn)
        return fn

    def on_ready(self, fn: Callable) -> Callable:
        return self._register("ready", fn)

    def on_user_transcript(self, fn: Callable) -> Callable:
        return self._register("user_transcript", fn)

    def on_agent_transcript(self, fn: Callable) -> Callable:
        return self._register("agent_transcript", fn)

    def on_agent_delta(self, fn: Callable) -> Callable:
        return self._register("agent_delta", fn)

    def on_agent_audio(self, fn: Callable) -> Callable:
        return self._register("agent_audio", fn)

    def on_error(self, fn: Callable) -> Callable:
        return self._register("error", fn)

    async def __aenter__(self) -> "AgentConnection":
        # Mint a short-lived session token instead of letting connect fall back to
        # the raw api_key bearer; a caller-supplied token= skips the mint.
        token = self._token or (await self._client.tokens.create()).token
        self._session = await self._client.sessions.connect(
            url=self._url, token=token, auto_resume=self._auto_resume
        )
        await self._session.__aenter__()
        await self._session.update(agent_id=self._agent_id)
        return self

    async def __aexit__(self, *exc_info) -> None:
        await self.aclose()

    async def _open_sink(self) -> None:
        # PlaybackSink is an async context manager whose __aenter__ starts the
        # PortAudio output stream; handle() alone only
        # buffers decoded PCM, so an unopened sink is silence. Entered by hand
        # rather than with `async with` because the close is latched and shared
        # with aclose(), so run() must not own the exit half of the block.
        if self._sink is not None:
            await self._sink.__aenter__()

    async def _close_sink(self) -> None:
        if self._sink is not None and not self._sink_closed:
            self._sink_closed = True
            await self._sink.aclose()

    def _start_mic(self) -> None:
        self._mic = asyncio.create_task(microphone_stream(self._session))
        # A bare task's exception is never retrieved, which turned a dead mic
        # into a caller talking to an agent that could not hear them, with no
        # error anywhere. Claim the outcome instead.
        self._mic.add_done_callback(self._on_mic_done)

    def _on_mic_done(self, task: asyncio.Task) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc is None:
            return
        self._remember_mic_error(exc)
        run_task = self._run_task
        if run_task is not None and not run_task.done():
            # run() is parked on the socket and would otherwise wait out the
            # whole call before noticing; break it out so it can report.
            run_task.cancel()

    def _remember_mic_error(self, exc: BaseException) -> None:
        if self._mic_error is None:
            self._mic_error = exc

    def _mic_failure(self) -> DeviceAudioError:
        return DeviceAudioError(
            "microphone capture stopped, so the agent is no longer receiving "
            f"any caller audio: {self._mic_error}. The session was ended rather "
            "than left running, because an agent that cannot hear the caller "
            "will keep talking while missing everything they say. This is "
            "almost always a missing, unplugged or already-busy input device: "
            "connect an input device and close whatever else is holding it, "
            "then reconnect. To drive a session without device capture, "
            "construct AgentConnection(audio=False) and feed audio yourself "
            "with session.send_audio()."
        )

    async def _cancel_mic(self) -> None:
        if self._mic is None:
            return
        mic = self._mic
        self._mic = None
        mic.cancel()
        # Drain so a cancelled OR already-failed mic task (e.g. no pyaudio) never
        # leaks an unretrieved exception. A real failure is kept to be reported;
        # only our own cancellation is discarded.
        try:
            await mic
        except asyncio.CancelledError:
            pass
        except BaseException as exc:
            self._remember_mic_error(exc)

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._cancel_mic()
        await self._close_sink()
        if self._session is not None:
            try:
                await self._session.end()
            except (ConnectionClosed, RealtimeError):
                # Best-effort: the server closes the socket right after
                # session.ended, so end() can fire at an already-dead socket —
                # don't let that mask the real teardown reason.
                pass
            await self._session.__aexit__(None, None, None)

    async def run(self) -> None:
        # AgentConnection owns the ONE async-for over the socket (single-reader
        # invariant); the mic pump is send-only and never reads it.
        self._run_task = asyncio.current_task()
        try:
            if self._sink is not None:
                # Both halves of device audio start here so that a failure in
                # either one lands inside the try and gets torn down below.
                await self._open_sink()
                self._start_mic()
            async for event in self._session:
                if self._sink is not None:
                    # Playback + barge-in are delegated to the sink's single home.
                    self._sink.handle(event)
                elif isinstance(event, ReplyAudio):
                    await self._fan_out("agent_audio", event)
                if isinstance(event, ToolCall):
                    # Every tool on a declaration is served over HTTPS by the
                    # process that declared it, so the platform runs tools itself
                    # and this client is never asked. Answer a stray one honestly
                    # rather than letting the call wait out the tool's timeout.
                    await self._session.send_tool_result(
                        event.call_id,
                        f"tool {event.name!r} is not served by this client; declare it "
                        f"on the agent so the platform calls it over HTTPS",
                        is_error=True,
                    )
                await self._dispatch(event)
                if isinstance(event, SessionEnded):
                    break
        except asyncio.CancelledError:
            # A dead mic cancels this task to break the async-for out of its
            # wait on the socket. That is the mic's failure to report below, not
            # a caller cancellation to propagate.
            if self._mic_error is None:
                raise
            task = asyncio.current_task()
            if task is not None:
                # Swallowing a cancellation without undoing it leaves the count
                # raised and confuses any enclosing timeout scope.
                task.uncancel()
        finally:
            # Cleared first so a late mic failure cannot cancel us while we are
            # already tearing down.
            self._run_task = None
            await self._cancel_mic()
            await self._close_sink()
        if self._mic_error is not None:
            raise self._mic_failure() from self._mic_error

    async def _dispatch(self, event: Any) -> None:
        if isinstance(event, TranscriptUser):
            await self._fan_out("user_transcript", event.text)
        elif isinstance(event, TranscriptAgent):
            await self._fan_out("agent_transcript", event.text)
        elif isinstance(event, TranscriptAgentDelta):
            await self._fan_out("agent_delta", event.delta)
        elif isinstance(event, SessionReady):
            await self._fan_out("ready", event)
        elif isinstance(event, SessionError):
            await self._fan_out("error", event)

    async def _fan_out(self, slot: str, value: Any) -> None:
        for fn in self._callbacks[slot]:
            await _maybe_await(fn(value))

    async def say(self, text: str) -> None:
        await self._session.send_message(text)

    @property
    def session(self) -> AsyncRealtimeSession:
        return self._session
