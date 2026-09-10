import asyncio
import collections
from typing import Optional

import websockets.exceptions

from ._exceptions import AssemblyAIAgentsError, RealtimeError
from .audio import base64_to_pcm
from .models.ws import (
    InputSpeechStarted,
    ReplyAudio,
    ReplyDone,
    ReplyStarted,
)


class DeviceAudioNotInstalledError(ImportError):
    def __init__(
        self,
        msg=(
            "You must install the audio extra to use device I/O. "
            'Run `pip install "assemblyai-agents[audio]"`. '
            "Before installing, install the PortAudio system library: "
            "`apt install portaudio19-dev` (Debian/Ubuntu) or "
            "`brew install portaudio` (macOS)."
        ),
        *args,
        **kwargs,
    ):
        super().__init__(msg, *args, **kwargs)


class DeviceAudioError(AssemblyAIAgentsError):
    pass


def _import_pyaudio():
    try:
        import pyaudio
    except ImportError as exc:
        raise DeviceAudioNotInstalledError from exc
    return pyaudio


async def microphone_stream(
    session,
    *,
    sample_rate: int = 24000,
    frames_per_buffer: int = 480,
    device: Optional[int] = None,
) -> None:
    pyaudio = _import_pyaudio()
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[bytes] = asyncio.Queue()

    def _on_capture(in_data, frame_count, time_info, status):
        # PortAudio's capture thread is NOT the event-loop thread; hand the buffer
        # off via call_soon_threadsafe. RuntimeError fires only if the loop is
        # already closed during shutdown — drop the trailing buffer, don't crash.
        try:
            loop.call_soon_threadsafe(queue.put_nowait, in_data)
        except RuntimeError:
            pass
        return (None, pyaudio.paContinue)

    handle = pyaudio.PyAudio()
    try:
        stream = handle.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=sample_rate,
            input=True,
            frames_per_buffer=frames_per_buffer,
            input_device_index=device,
            stream_callback=_on_capture,
        )
    except OSError as exc:
        handle.terminate()
        raise DeviceAudioError(f"failed to open input audio device: {exc}") from exc

    try:
        while True:
            raw_pcm = await queue.get()
            try:
                await session.send_audio(raw_pcm)
            except (websockets.exceptions.ConnectionClosed, RealtimeError):
                # auto_resume swaps the socket under us; an in-flight send can
                # raise here. Drop this chunk and keep capturing — the pump must
                # survive the swap, not die for the rest of the call.
                continue
    finally:
        stream.stop_stream()
        stream.close()
        handle.terminate()


class PlaybackSink:
    def __init__(
        self,
        *,
        sample_rate: int = 24000,
        frames_per_buffer: int = 480,
        device: Optional[int] = None,
    ) -> None:
        self._sample_rate = sample_rate
        self._frames_per_buffer = frames_per_buffer
        self._device = device
        # GIL-atomic deque, NOT an asyncio.Queue: the PortAudio output callback
        # reads it from its own thread, where asyncio.Queue is unsafe. deque
        # append/popleft/clear are atomic under the GIL.
        self._buffer: "collections.deque[bytes]" = collections.deque()
        self._interrupted: set[str] = set()
        self._active: Optional[str] = None
        self._pyaudio = None
        self._handle = None
        self._stream = None
        self._closed = False

    def _output_callback(self, in_data, frame_count, time_info, status):
        want = frame_count * 2
        out = bytearray()
        while len(out) < want and self._buffer:
            out.extend(self._buffer.popleft())
        if len(out) >= want:
            extra = bytes(out[want:])
            out = out[:want]
            if extra:
                self._buffer.appendleft(extra)
        else:
            out.extend(b"\x00" * (want - len(out)))
        return (bytes(out), self._pyaudio.paContinue)

    def _open(self) -> None:
        self._pyaudio = _import_pyaudio()
        self._handle = self._pyaudio.PyAudio()
        try:
            self._stream = self._handle.open(
                format=self._pyaudio.paInt16,
                channels=1,
                rate=self._sample_rate,
                output=True,
                frames_per_buffer=self._frames_per_buffer,
                output_device_index=self._device,
                stream_callback=self._output_callback,
            )
        except OSError as exc:
            self._handle.terminate()
            self._handle = None
            raise DeviceAudioError(
                f"failed to open output audio device: {exc}"
            ) from exc

    async def __aenter__(self) -> "PlaybackSink":
        self._open()
        return self

    async def __aexit__(self, *exc_info) -> None:
        await self.aclose()

    def handle(self, event) -> None:
        # The single home for reply-audio routing + barge-in. Touches only the
        # buffer/interrupted state + decode — no open PortAudio stream required,
        # so it is unit-testable without a sound card and reusable by hand-rollers.
        if isinstance(event, ReplyStarted):
            self._active = event.reply_id
        elif isinstance(event, ReplyAudio):
            self.write(event)
        elif isinstance(event, InputSpeechStarted):
            if self._active is not None:
                self.mark_interrupted(self._active)
            self.flush()
        elif isinstance(event, ReplyDone):
            # status is a Status enum at runtime; compare the value (Enum != str).
            status = getattr(event.status, "value", event.status)
            if status == "interrupted":
                self.mark_interrupted(event.reply_id)

    def write(self, event: ReplyAudio) -> None:
        if event.reply_id in self._interrupted:
            return
        self._buffer.append(base64_to_pcm(event.data))

    def flush(self) -> None:
        # Drop queued-but-unplayed audio on barge-in. pyaudio 0.2.14 has no
        # Stream.abort(), and stop_stream() DRAINS the buffered tail rather than
        # dropping it — so flush is deque-clear only; the callback emits silence
        # within one buffer and the stream stays open.
        self._buffer.clear()

    def mark_interrupted(self, reply_id: str) -> None:
        self._interrupted.add(reply_id)

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
        if self._handle is not None:
            self._handle.terminate()
