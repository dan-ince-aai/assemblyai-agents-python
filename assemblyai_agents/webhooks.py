import hashlib
import hmac
import json
import time

from ._exceptions import WebhookSignatureError, WebhookTimestampError


def verify(
    payload: bytes,
    signature_header: str,
    secret: str,
    *,
    tolerance: int = 300,
    now: int | None = None,
) -> dict:
    """Verify an AssemblyAI webhook signature and return the parsed event dict.

    ``payload`` MUST be the exact raw HTTP body bytes received over the wire,
    read BEFORE any JSON parse (e.g. ``await request.body()``): the MAC is
    computed over those bytes, so verifying a re-serialized dict (different key
    order or whitespace) recomputes a different MAC and spuriously rejects a
    valid delivery. ``signature_header`` is the ``X-AAI-Signature`` header value.
    Raises ``WebhookSignatureError`` (tamper / wrong secret / malformed header /
    non-JSON body) or ``WebhookTimestampError`` (timestamp outside ``tolerance``).
    """
    if not signature_header:
        raise WebhookSignatureError("missing or malformed signature header")

    fields = {}
    for pair in signature_header.split(","):
        key, sep, value = pair.partition("=")
        if sep:
            fields[key.strip()] = value.strip()

    raw_t = fields.get("t")
    v1 = fields.get("v1")
    if not raw_t or not v1:
        raise WebhookSignatureError("missing or malformed signature header")
    try:
        timestamp = int(raw_t)
    except ValueError:
        raise WebhookSignatureError("missing or malformed signature header") from None

    expected = hmac.new(
        secret.encode("utf-8"),
        f"{timestamp}.".encode("utf-8") + payload,
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(expected, v1):
        raise WebhookSignatureError("signature mismatch")

    now = now if now is not None else int(time.time())
    if abs(now - timestamp) > tolerance:
        raise WebhookTimestampError("timestamp outside tolerance")

    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        raise WebhookSignatureError("verified body is not valid JSON") from None
