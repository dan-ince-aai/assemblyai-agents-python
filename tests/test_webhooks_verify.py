import hashlib
import hmac
import json

import pytest
from assemblyai_agents import (
    WebhookSignatureError,
    WebhookTimestampError,
    WebhookVerificationError,
    webhooks,
)

# Server vector, lifted verbatim from the server-side signer's own test so this
# suite is a cross-implementation oracle.
SECRET = "whsec_" + "a" * 40
BODY = b'{"a":1,"event":"session.completed"}'
T = 1700000000
TOLERANCE = 300


def _mac(*, secret: str, body: bytes, timestamp: int) -> str:
    # The server signer's EXACT preimage: HMAC-SHA256 over
    # f"{t}.".encode() byte-concatenated with the raw body bytes, hex-encoded.
    return hmac.new(
        secret.encode("utf-8"),
        f"{timestamp}.".encode("utf-8") + body,
        hashlib.sha256,
    ).hexdigest()


def _header(*, secret: str = SECRET, body: bytes = BODY, timestamp: int = T) -> str:
    return f"t={timestamp},v1={_mac(secret=secret, body=body, timestamp=timestamp)}"


def test_verify_known_vector_returns_payload_dict():
    header = _header()
    result = webhooks.verify(BODY, header, SECRET, now=T)
    assert result == {"a": 1, "event": "session.completed"}
    assert result == json.loads(BODY)


def test_verify_rejects_tampered_payload():
    header = _header(body=BODY)
    tampered = BODY[:-1] + (b"X" if BODY[-1:] != b"X" else b"Y")
    with pytest.raises(WebhookSignatureError):
        webhooks.verify(tampered, header, SECRET, now=T)


def test_verify_rejects_wrong_secret():
    header = _header(secret=SECRET)
    with pytest.raises(WebhookSignatureError):
        webhooks.verify(BODY, header, "whsec_" + "b" * 40, now=T)


def test_verify_rejects_stale_timestamp():
    header = _header()
    with pytest.raises(WebhookTimestampError):
        webhooks.verify(
            BODY, header, SECRET, tolerance=TOLERANCE, now=T + TOLERANCE + 1
        )


def test_verify_rejects_future_timestamp():
    header = _header()
    with pytest.raises(WebhookTimestampError):
        webhooks.verify(
            BODY, header, SECRET, tolerance=TOLERANCE, now=T - TOLERANCE - 1
        )


def test_verify_accepts_within_tolerance():
    header = _header()
    result = webhooks.verify(
        BODY, header, SECRET, tolerance=TOLERANCE, now=T + TOLERANCE - 1
    )
    assert result == {"a": 1, "event": "session.completed"}


@pytest.mark.parametrize(
    "header",
    [
        "",
        f"v1={_mac(secret=SECRET, body=BODY, timestamp=T)}",  # missing t=
        f"t={T}",  # missing v1=
        f"t=notanint,v1={_mac(secret=SECRET, body=BODY, timestamp=T)}",  # non-int t
        "garbage",
        f"sha256={_mac(secret=SECRET, body=BODY, timestamp=T)}",  # legacy/unknown scheme
        f"t={T};v1={_mac(secret=SECRET, body=BODY, timestamp=T)}",  # wrong separator
        f"t=,v1={_mac(secret=SECRET, body=BODY, timestamp=T)}",  # empty t
    ],
)
def test_verify_rejects_malformed_header(header):
    # Fail closed: every malformed header raises, never accepts.
    with pytest.raises(WebhookSignatureError):
        webhooks.verify(BODY, header, SECRET, now=T)


def test_verify_uses_constant_time_compare(monkeypatch):
    header = _header()
    calls = []
    real_compare = hmac.compare_digest

    def _spy(a, b):
        calls.append((a, b))
        return real_compare(a, b)

    # Patch the module-bound name the verify body calls, NOT the global hmac
    # module, so this proves the v1 compare routes through compare_digest.
    monkeypatch.setattr(webhooks.hmac, "compare_digest", _spy)
    result = webhooks.verify(BODY, header, SECRET, now=T)
    assert result == {"a": 1, "event": "session.completed"}
    assert calls


def test_verify_ignores_unknown_keys():
    header = f"{_header()},v2=future,foo=bar"
    result = webhooks.verify(BODY, header, SECRET, now=T)
    assert result == {"a": 1, "event": "session.completed"}


def test_verify_does_not_leak_secret_in_exception():
    expected_mac = _mac(secret=SECRET, body=BODY, timestamp=T)

    # (1) bad-MAC failure -> WebhookSignatureError
    bad_header = f"t={T},v1={'0' * 64}"
    with pytest.raises(WebhookSignatureError) as sig_exc:
        webhooks.verify(BODY, bad_header, SECRET, now=T)
    sig_text = str(sig_exc.value) + repr(sig_exc.value)
    assert SECRET not in sig_text
    assert expected_mac not in sig_text
    assert "0" * 64 not in sig_text

    # (2) stale-but-valid-MAC failure -> WebhookTimestampError
    good_header = _header()
    with pytest.raises(WebhookTimestampError) as ts_exc:
        webhooks.verify(
            BODY, good_header, SECRET, tolerance=TOLERANCE, now=T + TOLERANCE + 1
        )
    ts_text = str(ts_exc.value) + repr(ts_exc.value)
    assert SECRET not in ts_text
    assert expected_mac not in ts_text

    # Both subclass the catch-all base so a caller can wrap one except.
    assert issubclass(WebhookSignatureError, WebhookVerificationError)
    assert issubclass(WebhookTimestampError, WebhookVerificationError)
