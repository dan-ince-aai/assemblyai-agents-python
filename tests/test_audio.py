import importlib.util
import struct

import pytest
from assemblyai_agents.audio import (
    alaw_to_pcm16,
    base64_to_pcm,
    pcm16_to_alaw,
    pcm16_to_ulaw,
    pcm_to_base64,
    ulaw_to_pcm16,
)

FIXED_PCM = struct.pack("<8h", 0, 1, -1, 32767, -32768, 100, -100, 12345)

ORACLE_ULAW = bytes.fromhex("ffff7e8000f27297")
ORACLE_ALAW = bytes.fromhex("d5d555aa2ad353bd")
ORACLE_ULAW_DECODED = bytes.fromhex("00000000f8ff7c7d8482680098ff7c30")
ORACLE_ALAW_DECODED = bytes.fromhex("08000800f8ff007e0082680098ff0031")


def test_base64_roundtrip_exact():
    cases = [
        b"",
        b"\x01",
        struct.pack("<h", -12345),
        FIXED_PCM,
        bytes(range(256)) * 16,
    ]
    for raw in cases:
        encoded = pcm_to_base64(raw)
        assert isinstance(encoded, str)
        assert base64_to_pcm(encoded) == raw


def test_base64_decode_rejects_garbage():
    with pytest.raises(ValueError):
        base64_to_pcm("not valid base64 !!!@@@")


def test_ulaw_encode_matches_oracle():
    assert pcm16_to_ulaw(FIXED_PCM) == ORACLE_ULAW


def test_ulaw_decode_matches_oracle():
    assert ulaw_to_pcm16(ORACLE_ULAW) == ORACLE_ULAW_DECODED


def test_alaw_encode_matches_oracle():
    assert pcm16_to_alaw(FIXED_PCM) == ORACLE_ALAW


def test_alaw_decode_matches_oracle():
    assert alaw_to_pcm16(ORACLE_ALAW) == ORACLE_ALAW_DECODED


def test_audio_module_imports_no_audioop():
    spec = importlib.util.find_spec("assemblyai_agents.audio")
    assert spec is not None and spec.origin is not None
    with open(spec.origin, encoding="utf-8") as fh:
        source = fh.read()
    import_lines = [
        ln for ln in source.splitlines() if ln.lstrip().startswith(("import ", "from "))
    ]
    assert not any("audioop" in ln for ln in import_lines), (
        "audio.py must not import audioop (removed in 3.13); offending: "
        f"{[ln for ln in import_lines if 'audioop' in ln]}"
    )


def test_companding_roundtrip_bounded():
    samples = struct.unpack("<8h", FIXED_PCM)

    ulaw_recon = struct.unpack("<8h", ulaw_to_pcm16(pcm16_to_ulaw(FIXED_PCM)))
    alaw_recon = struct.unpack("<8h", alaw_to_pcm16(pcm16_to_alaw(FIXED_PCM)))

    assert ulaw_recon != samples
    assert alaw_recon != samples

    for orig, u, a in zip(samples, ulaw_recon, alaw_recon):
        bound = abs(orig) // 16 + 16
        assert abs(orig - u) <= bound, (orig, u, bound)
        assert abs(orig - a) <= bound, (orig, a, bound)


def test_odd_length_pcm_raises():
    for bad in (b"\x01", b"\x01\x02\x03"):
        with pytest.raises(ValueError):
            pcm16_to_ulaw(bad)
        with pytest.raises(ValueError):
            pcm16_to_alaw(bad)


def test_extremes_encode_decode():
    extremes = struct.pack("<3h", 0, 32767, -32768)

    u = pcm16_to_ulaw(extremes)
    a = pcm16_to_alaw(extremes)
    assert len(u) == 3
    assert len(a) == 3

    assert len(ulaw_to_pcm16(u)) == 6
    assert len(alaw_to_pcm16(a)) == 6

    u_recon = struct.unpack("<3h", ulaw_to_pcm16(u))
    a_recon = struct.unpack("<3h", alaw_to_pcm16(a))
    assert u_recon[1] > 0 and u_recon[2] < 0
    assert a_recon[1] > 0 and a_recon[2] < 0

    assert pcm16_to_ulaw(b"") == b""
    assert pcm16_to_alaw(b"") == b""
    assert ulaw_to_pcm16(b"") == b""
    assert alaw_to_pcm16(b"") == b""
    assert pcm_to_base64(b"") == ""
    assert base64_to_pcm("") == b""
