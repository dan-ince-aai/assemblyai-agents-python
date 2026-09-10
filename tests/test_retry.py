import pytest
from assemblyai_agents import APIError, ServerError

from .conftest import Recorder, SleepRecorder, err


def test_retry_503_then_200_two_attempts(
    make_client, recorder: Recorder, sleep_recorder
):
    client = make_client(
        [(503, err("auth_service_unavailable"), None), (200, {"ok": True}, None)],
        recorder,
    )

    body = client.request("GET", "/v1/agents")

    assert body == {"ok": True}
    assert recorder.count == 2  # exactly one retry then success


def test_retry_exhausts_after_max_retries_raises_typed(
    make_client, recorder: Recorder, sleep_recorder
):
    client = make_client(
        [(503, err("auth_service_unavailable"), None)], recorder, max_retries=3
    )

    with pytest.raises(ServerError) as exc_info:
        client.request("GET", "/v1/agents")

    assert recorder.count == 4  # 1 initial + 3 retries
    assert exc_info.value.status == 503
    # A bare 503 with no parseable code still classifies by status, because the
    # class is chosen from the status and never from the (absent) code.
    recorder2 = Recorder()
    bare = make_client([(503, None, None)], recorder2, max_retries=3)
    with pytest.raises(APIError) as bare_exc:
        bare.request("GET", "/v1/agents")
    assert recorder2.count == 4
    assert bare_exc.value.status == 503


def test_retry_on_408_and_429(make_client, sleep_recorder):
    for status, code in ((408, None), (429, None)):
        recorder = Recorder()
        client = make_client(
            [(status, code, None), (200, {"ok": True}, None)], recorder
        )
        body = client.request("GET", "/v1/agents")
        assert body == {"ok": True}, f"{status} did not retry to success"
        msg = f"{status} expected 2 attempts, got {recorder.count}"
        assert recorder.count == 2, msg


def test_backoff_bounds_monotonic_capped_with_jitter(
    make_client, recorder: Recorder, sleep_recorder: SleepRecorder
):
    # 5xx every time, 5 retries, so the bound sequence reaches the cap.
    client = make_client(
        [(503, err("auth_service_unavailable"), None)], recorder, max_retries=5
    )

    with pytest.raises(ServerError):
        client.request("GET", "/v1/agents")

    # One sleep per retry (not after the final failed attempt): 5 retries.
    bounds = sleep_recorder.bounds
    assert len(bounds) == 5, bounds
    # base=0.5, cap=30, upper(n) = min(30, 0.5 * 2**n) for n = 0..4.
    expected = [min(30.0, 0.5 * (2**n)) for n in range(5)]
    assert bounds == pytest.approx(expected)
    # Monotonic non-decreasing and capped at 30s.
    assert all(b <= 30.0 for b in bounds)
    assert all(bounds[i] <= bounds[i + 1] for i in range(len(bounds) - 1))


def test_retry_after_header_honored_as_lower_bound(
    make_client, recorder: Recorder, sleep_recorder: SleepRecorder
):
    # Retry-After of 12s on the first 503; the jittered upper bound for the
    # first retry is min(30, 0.5*2**0) = 0.5, so the SDK must sleep >= 12.
    client = make_client(
        [
            (503, err("auth_service_unavailable"), {"Retry-After": "12"}),
            (200, {"ok": True}, None),
        ],
        recorder,
    )

    body = client.request("GET", "/v1/agents")

    assert body == {"ok": True}
    assert recorder.count == 2
    assert sleep_recorder.bounds[0] >= 12.0
    # A malformed Retry-After is ignored (falls back to pure jitter bound).
    recorder2 = Recorder()
    sleep2 = sleep_recorder
    sleep2.bounds.clear()
    client2 = make_client(
        [
            (503, err("auth_service_unavailable"), {"Retry-After": "not-a-number"}),
            (200, {"ok": True}, None),
        ],
        recorder2,
    )
    client2.request("GET", "/v1/agents")
    assert sleep2.bounds[0] == pytest.approx(0.5)


def test_retry_after_far_future_date_is_clamped(
    make_client, recorder: Recorder, sleep_recorder: SleepRecorder
):
    from assemblyai_agents._retry import _RETRY_AFTER_CAP

    # An HTTP-date years in the future would otherwise park the call for the
    # full delta; the SDK must clamp the wait to _RETRY_AFTER_CAP.
    far_future = "Wed, 01 Jan 2125 00:00:00 GMT"
    client = make_client(
        [
            (503, err("auth_service_unavailable"), {"Retry-After": far_future}),
            (200, {"ok": True}, None),
        ],
        recorder,
    )

    body = client.request("GET", "/v1/agents")

    assert body == {"ok": True}
    assert recorder.count == 2
    assert len(sleep_recorder.bounds) == 1
    assert sleep_recorder.bounds[0] <= _RETRY_AFTER_CAP


def test_no_retry_on_4xx_domain_error(make_client, sleep_recorder):
    from assemblyai_agents import (
        AuthenticationError,
        NotFoundError,
        ValidationError,
    )

    cases = [
        (404, err("agent_not_found"), NotFoundError),
        (
            422,
            err("validation_error", errors=[{"message": "x", "param": "n"}]),
            ValidationError,
        ),
        (401, err("unauthorized"), AuthenticationError),
    ]
    for status, body, exc_type in cases:
        recorder = Recorder()
        client = make_client([(status, body, None)], recorder)
        with pytest.raises(exc_type):
            client.request("GET", "/v1/agents")
        assert recorder.count == 1, f"{status} {body['code']} must not retry"
