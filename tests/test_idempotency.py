import uuid

import pytest
from assemblyai_agents import ConflictError, ValidationError

from .conftest import Recorder, err

HEADER = "Idempotency-Key"


def _is_uuid4(value: str) -> bool:
    parsed = uuid.UUID(value)
    return parsed.version == 4


def test_idempotency_key_reused_byte_identical_across_retries(
    make_client, recorder: Recorder, sleep_recorder
):
    client = make_client(
        [
            (503, err("auth_service_unavailable"), None),
            (503, err("auth_service_unavailable"), None),
            (201, {"id": "call_1"}, None),
        ],
        recorder,
    )

    client.request("POST", "/v1/calls", json={"agent_id": "a1"}, idempotent=True)

    assert recorder.count == 3
    keys = recorder.headers(HEADER)
    assert all(k is not None for k in keys), keys
    assert len(set(keys)) == 1, f"key changed across retries: {keys}"
    assert _is_uuid4(keys[0])


def test_idempotency_distinct_keys_across_logical_requests(
    make_client, recorder: Recorder
):
    client = make_client([(201, {"id": "x"}, None)], recorder)

    client.request("POST", "/v1/calls", json={"agent_id": "a"}, idempotent=True)
    client.request("POST", "/v1/calls", json={"agent_id": "b"}, idempotent=True)

    keys = recorder.headers(HEADER)
    assert len(keys) == 2
    assert keys[0] is not None and keys[1] is not None
    assert keys[0] != keys[1], "two logical requests must mint distinct keys"


def test_no_idempotency_key_when_not_idempotent(make_client, recorder: Recorder):
    client = make_client([(200, {"ok": True}, None)], recorder)

    # idempotent defaults to False
    client.request("POST", "/v1/agents", json={"name": "x"})

    assert recorder.header(HEADER) is None


def test_caller_idempotency_key_disables_auto_mint(make_client, recorder: Recorder):
    caller_key = "caller-supplied-key-123"
    client = make_client([(201, {"id": "x"}, None)], recorder)

    client.request(
        "POST",
        "/v1/calls",
        json={"agent_id": "a"},
        idempotent=True,
        headers={HEADER: caller_key},
    )

    # Caller's verbatim key is sent; the SDK does not mint over it.
    assert recorder.header(HEADER) == caller_key


def test_409_in_progress_retried_same_key(
    make_client, recorder: Recorder, sleep_recorder
):
    client = make_client(
        [(409, err("idempotency_in_progress"), None), (201, {"id": "c1"}, None)],
        recorder,
    )

    body = client.request("POST", "/v1/calls", json={"agent_id": "a"}, idempotent=True)

    assert body == {"id": "c1"}
    assert recorder.count == 2
    keys = recorder.headers(HEADER)
    assert keys[0] is not None and keys[0] == keys[1], keys


def test_422_key_reuse_terminal_single_attempt(
    make_client, recorder: Recorder, sleep_recorder
):
    client = make_client([(422, err("idempotency_key_reuse"), None)], recorder)

    with pytest.raises(ValidationError):
        client.request("POST", "/v1/calls", json={"agent_id": "a"}, idempotent=True)

    assert recorder.count == 1, "key-reuse 422 is terminal — never retried"


def test_409_phone_number_conflict_not_retried(
    make_client, recorder: Recorder, sleep_recorder
):
    client = make_client([(409, err("phone_number_conflict"), None)], recorder)

    with pytest.raises(ConflictError):
        client.request(
            "POST", "/v1/phone-numbers", json={"number": "+1555"}, idempotent=True
        )

    assert recorder.count == 1, "only idempotency_in_progress 409 is retryable"
