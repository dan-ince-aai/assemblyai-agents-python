import pytest
from assemblyai_agents import (
    APIError,
    ErrorCode,
    NotFoundError,
    ServerError,
    ValidationError,
)

from .conftest import Recorder, err


def test_code_maps_to_typed_exception_with_request_id(make_client, recorder: Recorder):
    client = make_client(
        [(404, err("agent_not_found", request_id="req-1"), None)], recorder
    )

    with pytest.raises(NotFoundError) as exc_info:
        client.request("GET", "/v1/agents/nope")

    exc = exc_info.value
    assert exc.status == 404
    assert exc.code == "agent_not_found"
    assert exc.request_id == "req-1"


def test_validation_error_populates_errors_list(make_client, recorder: Recorder):
    body = err(
        "validation_error",
        request_id="req-2",
        errors=[{"message": "x", "param": "name"}],
    )
    client = make_client([(422, body, None)], recorder)

    with pytest.raises(ValidationError) as exc_info:
        client.request("POST", "/v1/agents", json={})

    exc = exc_info.value
    assert exc.status == 422
    assert exc.code == "validation_error"
    assert exc.request_id == "req-2"
    assert exc.errors == [{"message": "x", "param": "name"}]


def test_unknown_code_on_known_status_still_maps_to_status_class(
    make_client, recorder: Recorder
):
    # The whole point of keying on status rather than code: a newer server can
    # invent a code this SDK has never heard of, and the caller who wrote
    # `except NotFoundError` must still catch it. The unrecognised code is
    # preserved verbatim on `.code` for the caller to inspect.
    client = make_client(
        [(404, err("some_future_code", request_id="req-3"), None)], recorder
    )

    with pytest.raises(NotFoundError) as exc_info:
        client.request("GET", "/v1/agents/nope")

    exc = exc_info.value
    assert type(exc) is NotFoundError
    assert exc.status == 404
    assert exc.code == "some_future_code"
    assert exc.request_id == "req-3"


def test_unmapped_status_degrades_to_base_apierror(make_client, recorder: Recorder):
    # 429 is retried by _retry but the server never emits it, so no subclass
    # claims it. An unmapped status must degrade to the base APIError rather
    # than guess a class from the status family.
    client = make_client(
        [(429, err("some_rate_limit_code", request_id="req-4"), None)],
        recorder,
        max_retries=0,
    )

    with pytest.raises(APIError) as exc_info:
        client.request("GET", "/v1/agents")

    exc = exc_info.value
    assert type(exc) is APIError
    assert exc.status == 429
    assert exc.code == "some_rate_limit_code"


def test_unparseable_edge_502_is_server_error_request_id_from_header(
    make_client, recorder: Recorder, sleep_recorder
):
    # A non-JSON 502 from an edge/proxy: no parseable envelope, so there is no
    # code to key on at all. Status-keying still classifies it — this is the
    # case code-keying could never handle. request_id falls back to the
    # X-Request-Id header. Retries (502 is 5xx) then raises on exhaustion.
    client = make_client(
        [(502, b"<html>502 Bad Gateway</html>", {"X-Request-Id": "edge-req-9"})],
        recorder,
        max_retries=1,
    )

    with pytest.raises(ServerError) as exc_info:
        client.request("GET", "/v1/agents")

    exc = exc_info.value
    assert type(exc) is ServerError
    assert exc.status == 502
    assert exc.code is None
    assert exc.request_id == "edge-req-9"
    assert recorder.count == 2  # 1 + 1 retry, both 502


def test_decoded_envelope_without_message_is_not_called_unparseable(
    make_client, recorder: Recorder
):
    # A well-formed JSON envelope that simply omits `message`. The body parsed
    # fine, so the diagnostic must say the API supplied no reason rather than
    # assert the body was empty or was not JSON.
    body = err("agent_not_found", request_id="req-7")
    del body["message"]
    client = make_client([(404, body, None)], recorder)

    with pytest.raises(NotFoundError) as exc_info:
        client.request("GET", "/v1/agents/nope")

    message = str(exc_info.value)
    assert "no `message`" in message
    assert "not JSON" not in message
    assert "`.code`" in message
    assert "req-7" in message


def test_unreadable_body_still_says_the_body_could_not_be_read(
    make_client, recorder: Recorder
):
    # The other half of the split: a genuinely unreadable body keeps the
    # edge-proxy diagnosis and the pointer at `.raw.content`.
    client = make_client(
        [(404, b"<html>404 Not Found</html>", {"X-Request-Id": "edge-req-10"})],
        recorder,
    )

    with pytest.raises(NotFoundError) as exc_info:
        client.request("GET", "/v1/agents/nope")

    message = str(exc_info.value)
    assert "not JSON" in message
    assert "`.raw.content`" in message
    assert "edge-req-10" in message


def test_error_code_constants_compare_as_plain_strings(make_client, recorder: Recorder):
    # ErrorCode is deliberately a plain-string constants class, so a known code
    # compares equal to its constant and an unknown code simply compares false.
    # Neither comparison raises, which is what makes forward compatibility work.
    client = make_client(
        [(404, err("agent_not_found", request_id="req-5"), None)], recorder
    )
    with pytest.raises(NotFoundError) as exc_info:
        client.request("GET", "/v1/agents/nope")

    exc = exc_info.value
    assert exc.code == ErrorCode.AGENT_NOT_FOUND
    assert exc.code != ErrorCode.CALL_NOT_FOUND

    recorder2 = Recorder()
    client2 = make_client(
        [(404, err("some_future_code", request_id="req-6"), None)], recorder2
    )
    with pytest.raises(NotFoundError) as exc_info2:
        client2.request("GET", "/v1/agents/nope")

    # An unheard-of code is still a str, so every comparison against every
    # ErrorCode constant is simply False rather than an exception or a
    # type error.
    unknown = exc_info2.value
    assert unknown.code == "some_future_code"
    assert unknown.code != ErrorCode.AGENT_NOT_FOUND
    assert unknown.code != ErrorCode.INTERNAL_ERROR
