import asyncio
import email.utils
import random
import time
from datetime import timezone
from typing import Optional

BASE = 0.5
CAP = 30.0
# A skewed or far-future Retry-After date must not be able to park the call for
# minutes; honor up to a minute, clamp anything beyond.
_RETRY_AFTER_CAP = 60.0

RETRYABLE_STATUSES = {408, 429}
RETRYABLE_409_CODE = "idempotency_in_progress"


def is_retryable_status(status: int, code: Optional[str]) -> bool:
    if status in RETRYABLE_STATUSES:
        return True
    if 500 <= status <= 599:
        return True
    if status == 409 and code == RETRYABLE_409_CODE:
        return True
    return False


def backoff_bound(retry_index: int) -> float:
    return min(CAP, BASE * (2**retry_index))


def sleep_seconds(retry_index: int, retry_after: Optional[str]) -> float:
    jittered = random.uniform(0, backoff_bound(retry_index))
    floor = parse_retry_after(retry_after)
    if floor is None:
        return jittered
    return min(_RETRY_AFTER_CAP, max(floor, jittered))


def parse_retry_after(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    text = value.strip()
    if text.isdigit():
        return float(text)
    try:
        parsed = email.utils.parsedate_to_datetime(text)
    except (TypeError, ValueError):
        return None
    if parsed is None:
        return None
    # An HTTP-date with no zone is UTC by spec; a naive datetime would otherwise
    # be read in local time and skew the computed delta.
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    delta = parsed.timestamp() - time.time()
    return delta if delta > 0 else 0.0


def sleep_sync(seconds: float) -> None:
    time.sleep(seconds)


async def sleep_async(seconds: float) -> None:
    await asyncio.sleep(seconds)
