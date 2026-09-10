import base64
import struct

_ULAW_BIAS = 0x84
_ULAW_CLIP = 8159
_ULAW_SEG_END = (0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF)
_ALAW_SEG_END = (0x1F, 0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF)


def _segment(value, seg_end):
    for index, end in enumerate(seg_end):
        if value <= end:
            return index
    return len(seg_end)


def pcm_to_base64(pcm):
    return base64.b64encode(pcm).decode("ascii")


def base64_to_pcm(data):
    try:
        return base64.b64decode(data, validate=True)
    except ValueError as exc:
        raise ValueError("invalid base64 audio payload") from exc


def pcm16_to_ulaw(pcm):
    if len(pcm) % 2:
        raise ValueError("PCM16 buffer length must be even (2 bytes per sample)")
    out = bytearray(len(pcm) // 2)
    for index, (sample,) in enumerate(struct.iter_unpack("<h", pcm)):
        magnitude = sample >> 2
        if magnitude < 0:
            magnitude = -magnitude
            mask = 0x7F
        else:
            mask = 0xFF
        if magnitude > _ULAW_CLIP:
            magnitude = _ULAW_CLIP
        magnitude += _ULAW_BIAS >> 2
        segment = _segment(magnitude, _ULAW_SEG_END)
        if segment >= 8:
            out[index] = 0x7F ^ mask
        else:
            out[index] = ((segment << 4) | ((magnitude >> (segment + 1)) & 0x0F)) ^ mask
    return bytes(out)


def ulaw_to_pcm16(ulaw):
    out = bytearray(len(ulaw) * 2)
    for index, byte in enumerate(ulaw):
        u = ~byte & 0xFF
        magnitude = ((u & 0x0F) << 3) + _ULAW_BIAS
        magnitude <<= (u & 0x70) >> 4
        sample = _ULAW_BIAS - magnitude if u & 0x80 else magnitude - _ULAW_BIAS
        struct.pack_into("<h", out, index * 2, sample)
    return bytes(out)


def pcm16_to_alaw(pcm):
    if len(pcm) % 2:
        raise ValueError("PCM16 buffer length must be even (2 bytes per sample)")
    out = bytearray(len(pcm) // 2)
    for index, (sample,) in enumerate(struct.iter_unpack("<h", pcm)):
        magnitude = sample >> 3
        if magnitude >= 0:
            mask = 0xD5
        else:
            mask = 0x55
            magnitude = -magnitude - 1
        segment = _segment(magnitude, _ALAW_SEG_END)
        if segment >= 8:
            out[index] = 0x7F ^ mask
        else:
            aval = segment << 4
            if segment < 2:
                aval |= (magnitude >> 1) & 0x0F
            else:
                aval |= (magnitude >> segment) & 0x0F
            out[index] = aval ^ mask
    return bytes(out)


def alaw_to_pcm16(alaw):
    out = bytearray(len(alaw) * 2)
    for index, byte in enumerate(alaw):
        a = byte ^ 0x55
        magnitude = (a & 0x0F) << 4
        segment = (a & 0x70) >> 4
        if segment == 0:
            magnitude += 8
        elif segment == 1:
            magnitude += 0x108
        else:
            magnitude = (magnitude + 0x108) << (segment - 1)
        struct.pack_into("<h", out, index * 2, magnitude if a & 0x80 else -magnitude)
    return bytes(out)
