import pytest


# This is the ONLY test that imports REAL pyaudio. It does NOT call pyaudio.PyAudio()
# (device enumeration can crash on a machine with no sound card) — it inspects the
# Stream CLASS surface only. The pytest.skip on ImportError keeps it from
# false-failing where the PortAudio library is not installed.
def test_real_pyaudio_stream_surface():
    try:
        import pyaudio  # real C-ext; tier-2 local-only guard
    except ImportError:
        pytest.skip(
            "real pyaudio cannot import in this environment (no portaudio)"
        )

    # Inspect the class surface only — never construct PyAudio() (no enumeration).
    assert hasattr(pyaudio.Stream, "stop_stream")
    assert hasattr(pyaudio.Stream, "write")
    assert hasattr(pyaudio.Stream, "read")
    assert hasattr(pyaudio.Stream, "close")
    assert not hasattr(pyaudio.Stream, "abort"), (
        "pyaudio 0.2.14 Stream has no abort(); flush must be deque-clear, not abort()"
    )
