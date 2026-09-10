import asyncio
import base64
import importlib
import sys
from pathlib import Path

import pytest

# tomllib is stdlib on the SDK's required-python (>=3.11; pyproject.toml:10).
import tomllib

# audio_io must NEVER import real pyaudio at module scope: importing the
# module here, with no pyaudio in sys.modules, is itself part of the lazy-import proof.
from assemblyai_agents import audio_io
from assemblyai_agents.models.ws import ReplyAudio
from websockets.exceptions import ConnectionClosedError
from websockets.frames import Close



# --- pyaudio module-surface constants we model faithfully (pyaudio 0.2.14) -----
paInt16 = 8
paContinue = 0
paComplete = 1


class _AbortGuard:
    """Descriptor that records and REJECTS any access to `Stream.abort`.

    pyaudio 0.2.14's Stream has NO `abort()` (de-risk §load-bearing finding 1). A
    permissive MagicMock would auto-stub it and false-green an impl that called
    abort() and would crash on a real device. This descriptor makes the fake faithful:
    touching `stream.abort` records the access and raises AttributeError, exactly as
    the real wrapper would.
    """

    def __init__(self):
        self.accessed = False

    def __get__(self, obj, objtype=None):
        self.accessed = True
        raise AttributeError("'Stream' object has no attribute 'abort'")


class FakeStream:
    """Models pyaudio 0.2.14's Stream surface (NO `abort`).

    Callback mode: `open(..., stream_callback=cb)` stores `cb` here so a test can
    invoke it to simulate PortAudio pulling (output) / pushing (input) buffers. The
    blocking-mode `read`/`write` are present (the real wrapper has them) and recorded.
    """

    abort = _AbortGuard()  # class attr: access raises + records (see _AbortGuard)

    def __init__(self, *, stream_callback=None, frames_per_buffer=480, **kwargs):
        self.stream_callback = stream_callback
        self.frames_per_buffer = frames_per_buffer
        self.open_kwargs = kwargs
        self.writes: list[bytes] = []
        self.started = False
        self.start_calls = 0
        self.stop_calls = 0
        self.close_calls = 0
        self._active = True

    def start_stream(self):
        self.start_calls += 1
        self.started = True

    def stop_stream(self):
        self.stop_calls += 1
        self.started = False

    def write(self, data):
        self.writes.append(bytes(data))

    def read(self, num_frames, exception_on_overflow=False):
        return b"\x00\x00" * num_frames

    def is_active(self):
        return self._active

    def close(self):
        self.close_calls += 1
        self._active = False


class FakePyAudio:
    """Models pyaudio's module + PyAudio() handle surface.

    `open()` builds a FakeStream and records it on `last_stream`. To exercise the
    no-device path (criterion 8) a test sets `open_raises` to an OSError before opening.
    """

    paInt16 = paInt16
    paContinue = paContinue
    paComplete = paComplete

    def __init__(self):
        self.streams: list[FakeStream] = []
        self.terminate_calls = 0
        self.open_raises: BaseException | None = None

    # module-level `pyaudio.PyAudio` is a class; the helper does pyaudio.PyAudio().
    def PyAudio(self):  # noqa: N802 - mirror pyaudio's public name
        return self

    def open(self, **kwargs):
        if self.open_raises is not None:
            raise self.open_raises
        stream = FakeStream(**kwargs)
        stream.start_stream()
        self.streams.append(stream)
        return stream

    def terminate(self):
        self.terminate_calls += 1

    def get_default_input_device_info(self):
        return {"index": 0, "name": "fake-in", "maxInputChannels": 1}

    def get_default_output_device_info(self):
        return {"index": 0, "name": "fake-out", "maxOutputChannels": 1}

    @property
    def last_stream(self) -> FakeStream:
        return self.streams[-1]


def _install_fake_pyaudio(monkeypatch) -> FakePyAudio:
    fake = FakePyAudio()
    monkeypatch.setitem(sys.modules, "pyaudio", fake)
    return fake


def _dropped(msg: str = "socket swapped") -> ConnectionClosedError:
    # Real ConnectionClosed (subclass) — exactly what send_audio raises when the
    # auto-resume swap closes the socket mid-send; the pump's narrow catch must
    # survive it and continue (Risk 5).
    return ConnectionClosedError(Close(1006, msg), None)


class FakeSession:
    """A send-only fake of AsyncRealtimeSession for the mic pump.

    `send_audio` records raw bytes verbatim. `recv`/`__anext__` RAISE if ever touched —
    the mic pump is send-only and must NEVER read the socket (single-reader invariant,
    solution.md:10). `send_audio` can be scripted to raise on chosen call indices to
    exercise the auto-resume drop tolerance (Risk 5).
    """

    def __init__(self, *, raise_on: set[int] | None = None, exc=None):
        self.sent: list[bytes] = []
        self._raise_on = raise_on or set()
        self._exc = exc or _dropped("socket swapped")
        self._call = 0

    async def send_audio(self, audio: bytes) -> None:
        idx = self._call
        self._call += 1
        if idx in self._raise_on:
            raise self._exc
        self.sent.append(bytes(audio))

    async def recv(self):
        raise AssertionError("mic pump must not read the socket (recv)")

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise AssertionError("mic pump must not iterate the socket (__anext__)")


class ImportFailureMocker:
    """meta_path finder that makes `import pyaudio` raise ImportError."""

    def __init__(self, module: str):
        self.module = module

    def find_spec(self, fullname, path, target=None):
        if fullname == self.module:
            raise ImportError(f"mocked missing module: {fullname}")

    def __enter__(self):
        if self.module in sys.modules:
            del sys.modules[self.module]
        sys.meta_path.insert(0, self)
        return self

    def __exit__(self, exc_type, exc, tb):
        sys.meta_path.pop(0)


def _reply(reply_id: str, pcm: bytes) -> ReplyAudio:
    return ReplyAudio(reply_id=reply_id, data=base64.b64encode(pcm).decode())


async def _pump_loop(n: int = 5) -> None:
    """Yield control n times so background drain tasks get to run."""
    for _ in range(n):
        await asyncio.sleep(0)


async def test_mic_pump_forwards_raw_pcm(monkeypatch):
    fake = _install_fake_pyaudio(monkeypatch)
    session = FakeSession()  # recv/__anext__ raise if the pump ever reads the socket

    pump = asyncio.create_task(
        audio_io.microphone_stream(session, sample_rate=24000, frames_per_buffer=480)
    )
    await _pump_loop()  # let the pump open its callback-mode input stream

    stream = fake.last_stream
    assert stream.stream_callback is not None, "mic must run in callback mode"

    chunk_a = b"\x01\x02" * 240  # 480 frames of PCM16 (raw mic bytes)
    chunk_b = b"\x7f\x7f" * 240
    # Simulate PortAudio's capture thread handing buffers to the callback.
    cb = stream.stream_callback
    assert cb(chunk_a, 480, {}, 0)[1] == paContinue
    assert cb(chunk_b, 480, {}, 0)[1] == paContinue

    await _pump_loop()  # let the async drain task forward both chunks

    pump.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pump

    # Byte-identical, in order, NOT base64-re-encoded (send_audio encodes internally).
    assert session.sent == [chunk_a, chunk_b]
    # And the single-reader invariant held: FakeSession.recv/__anext__ would have
    # raised AssertionError out of the pump if it had read the socket.


async def test_playback_decodes_and_writes(monkeypatch):
    fake = _install_fake_pyaudio(monkeypatch)
    pcm_a = b"\x11\x22" * 240
    pcm_b = b"\x33\x44" * 240

    async with audio_io.PlaybackSink(sample_rate=24000, frames_per_buffer=480) as sink:
        stream = fake.last_stream
        assert stream.stream_callback is not None, "playback must run in callback mode"
        cb = stream.stream_callback

        sink.write(_reply("r1", pcm_a))
        sink.write(_reply("r1", pcm_b))

        # PortAudio pulls one buffer per callback; assert decoded PCM drains in order.
        out_a, flag_a = cb(None, 240, {}, 0)
        out_b, flag_b = cb(None, 240, {}, 0)

    assert out_a == pcm_a
    assert out_b == pcm_b
    assert flag_a == paContinue and flag_b == paContinue
    # The decode path is base64_to_pcm: a non-base64 payload would have raised here.


async def test_flush_drops_queued_audio_on_barge_in(monkeypatch):
    fake = _install_fake_pyaudio(monkeypatch)
    pcm_played = b"\x01\x01" * 240
    pcm_dropped = b"\x09\x09" * 240

    async with audio_io.PlaybackSink(frames_per_buffer=480) as sink:
        cb = fake.last_stream.stream_callback

        sink.write(_reply("r1", pcm_played))
        sink.write(_reply("r1", pcm_dropped))  # queued behind pcm_played

        # Pull the first chunk, then barge-in: flush before the second is pulled.
        out_first, _ = cb(None, 240, {}, 0)
        sink.flush()
        out_after, flag_after = cb(None, 240, {}, 0)

    assert out_first == pcm_played
    # Flushed chunk never plays: the post-flush callback yields silence, not pcm_dropped.
    assert out_after != pcm_dropped
    assert set(out_after) == {0}, "post-flush callback must emit silence"
    assert len(out_after) == 240 * 2  # paInt16 mono -> 2 bytes/frame
    assert flag_after == paContinue  # stream stays open


async def test_flush_keeps_stream_open_never_aborts(monkeypatch):
    fake = _install_fake_pyaudio(monkeypatch)

    async with audio_io.PlaybackSink(frames_per_buffer=480) as sink:
        stream = fake.last_stream
        sink.write(_reply("r1", b"\x05\x06" * 240))

        stop_before = stream.stop_calls
        close_before = stream.close_calls

        sink.flush()

        # flush must NOT tear down or drain the stream.
        assert stream.stop_calls == stop_before, "flush must not stop_stream()"
        assert stream.close_calls == close_before, "flush must not close()"
        assert stream.is_active(), "stream must stay open after flush"
        # abort() does not exist on pyaudio 0.2.14: the _AbortGuard records any access.
        assert type(stream).__dict__["abort"].accessed is False, (
            "flush must never touch Stream.abort (it does not exist on pyaudio 0.2.14)"
        )


async def test_interrupted_reply_id_dropped(monkeypatch):
    fake = _install_fake_pyaudio(monkeypatch)
    pcm_live = b"\x21\x22" * 240
    pcm_stale = b"\x31\x32" * 240

    async with audio_io.PlaybackSink(frames_per_buffer=480) as sink:
        cb = fake.last_stream.stream_callback

        sink.write(_reply("live", pcm_live))
        sink.mark_interrupted("stale")
        sink.write(_reply("stale", pcm_stale))  # must be dropped, not enqueued

        out_first, _ = cb(None, 240, {}, 0)
        out_second, flag = cb(None, 240, {}, 0)

    assert out_first == pcm_live
    # The interrupted reply_id's audio never reaches the device: queue holds only "live".
    assert out_second != pcm_stale
    assert set(out_second) == {0}, "interrupted reply audio must be dropped -> silence"
    assert flag == paContinue


async def test_mic_pump_survives_send_drop(monkeypatch):
    fake = _install_fake_pyaudio(monkeypatch)
    # First send_audio call raises the auto-resume socket-swap error; pump must survive.
    session = FakeSession(raise_on={0}, exc=_dropped("connection closed during swap"))

    pump = asyncio.create_task(
        audio_io.microphone_stream(session, frames_per_buffer=480)
    )
    await _pump_loop()
    cb = fake.last_stream.stream_callback

    dropped = b"\x01\x01" * 240  # this send raises -> chunk dropped
    survived = b"\x02\x02" * 240  # this one must still get through

    assert cb(dropped, 480, {}, 0)[1] == paContinue
    await _pump_loop()
    assert cb(survived, 480, {}, 0)[1] == paContinue
    await _pump_loop()

    assert not pump.done(), "pump must not die when send_audio raises ConnectionClosed"
    pump.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pump

    # The dropped chunk never landed; the post-drop chunk did. Pump kept capturing.
    assert survived in session.sent
    assert dropped not in session.sent


async def test_lazy_import_missing_raises_friendly(monkeypatch):
    # Drop any test-injected fake so the lazy import genuinely fails.
    monkeypatch.delitem(sys.modules, "pyaudio", raising=False)

    with ImportFailureMocker("pyaudio"):
        # The package and the module both import with NO pyaudio present: proves the
        # lazy import (pyaudio is never touched at module scope).
        pkg = importlib.import_module("assemblyai_agents")
        mod = importlib.import_module("assemblyai_agents.audio_io")
        importlib.reload(mod)
        assert pkg is not None and mod is not None
        assert "pyaudio" not in sys.modules

        # Touching a device-backed helper now raises the friendly install error.
        with pytest.raises(audio_io.DeviceAudioNotInstalledError):
            async with audio_io.PlaybackSink():
                pass

    assert issubclass(audio_io.DeviceAudioNotInstalledError, ImportError)


async def test_lifecycle_idempotent_and_no_device_safe(monkeypatch):
    fake = _install_fake_pyaudio(monkeypatch)

    # (a) idempotent teardown: stop -> close -> terminate exactly once across two acloses.
    sink = audio_io.PlaybackSink(frames_per_buffer=480)
    if hasattr(sink, "__aenter__"):
        await sink.__aenter__()
    stream = fake.last_stream

    await sink.aclose()
    await sink.aclose()  # second call must be a no-op, not an error

    assert stream.stop_calls == 1
    assert stream.close_calls == 1
    assert fake.terminate_calls == 1

    # (b) no-device: an OSError from open() surfaces as a clean error, not a raw OSError
    # carrying a raw PortAudio errno bubbling out unchanged.
    fake.open_raises = OSError(
        -9996, "Invalid output device (no default output device)"
    )
    with pytest.raises(Exception) as excinfo:
        async with audio_io.PlaybackSink(frames_per_buffer=480):
            pass
    # It must be a readable, intentional error — not the bare OSError re-raised verbatim.
    # (If the impl chooses a named AssemblyAIAgentsError subclass, tighten this assert.)
    assert not (type(excinfo.value) is OSError and excinfo.value.errno == -9996), (
        "raw PortAudio OSError must be wrapped in a clean error"
    )
    assert str(excinfo.value)  # has a human-readable message


def test_pyproject_declares_websockets_and_audio_extra():
    # Locate pyproject.toml relative to this test file (tests/ is a sibling of it),
    # independent of the working directory pytest was launched from.
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    assert pyproject.is_file(), f"pyproject.toml not found at {pyproject}"
    data = tomllib.loads(pyproject.read_text())

    core_deps = data["project"]["dependencies"]
    assert any(
        d.split()[0].lower().startswith("websockets")
        or d.lower().startswith("websockets")
        for d in core_deps
    ), (
        f"websockets must be a core dependency (clean-install import fix); got {core_deps}"
    )

    extras = data["project"]["optional-dependencies"]
    assert "audio" in extras, f"[audio] optional extra missing; got {list(extras)}"
    assert any("pyaudio" in d.lower() for d in extras["audio"]), (
        f"[audio] extra must declare pyaudio; got {extras['audio']}"
    )
