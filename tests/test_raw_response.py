import json

import httpx
import pytest
from assemblyai_agents import RawResponse, ResponseError

from .conftest import Recorder, err


def test_raw_response_does_not_raise_returns_exact_bytes(
    make_client, recorder: Recorder
):
    # Exact server bytes, formatted with non-canonical spacing, so a
    # re-serialization regression would change them.
    raw_bytes = b'{"code": "agent_not_found",  "request_id":"req-raw"}'
    client = make_client(
        [
            httpx.Response(
                status_code=404,
                content=raw_bytes,
                headers={"X-Request-Id": "req-raw"},
            )
        ],
        recorder,
    )

    resp = client.request_raw("GET", "/v1/agents/nope")

    assert isinstance(resp, RawResponse)
    assert resp.status_code == 404
    assert resp.content == raw_bytes  # byte-identical, NOT re-serialized
    assert resp.request_id == "req-raw"
    assert resp.json() == json.loads(raw_bytes)
    # Did NOT raise on a 4xx.
    assert recorder.count == 1


def test_raw_response_applies_retry_loop(
    make_client, recorder: Recorder, sleep_recorder
):
    client = make_client(
        [(503, err("auth_service_unavailable"), None), (200, {"ok": True}, None)],
        recorder,
    )

    resp = client.request_raw("GET", "/v1/agents")

    assert isinstance(resp, RawResponse)
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}
    assert recorder.count == 2  # retry loop applied before wrapping


def test_2xx_non_json_body_raises_response_error(make_client, recorder: Recorder):
    # A 200 whose body is not JSON: request() must surface a typed SDK error,
    # never let a bare json.JSONDecodeError escape.
    body = b"<html>not json</html>"
    client = make_client([httpx.Response(status_code=200, content=body)], recorder)

    with pytest.raises(ResponseError) as exc_info:
        client.request("GET", "/v1/agents")

    exc = exc_info.value
    assert exc.raw.status_code == 200
    assert exc.raw.content == body
    assert exc.request_id == exc.raw.request_id

    # The raw escape hatch on the same response does NOT raise — the bytes are
    # handed back untouched for the caller to inspect.
    recorder2 = Recorder()
    client2 = make_client([httpx.Response(status_code=200, content=body)], recorder2)
    resp = client2.request_raw("GET", "/v1/agents")
    assert isinstance(resp, RawResponse)
    assert resp.status_code == 200
    assert resp.content == body


def test_per_request_timeout_override(make_client, recorder: Recorder):
    # Default client timeout is 30.0; assert a per-request override reaches the
    # httpx request. httpx stamps the effective timeout on the request's
    # extensions["timeout"] dict (assumption R4).
    client = make_client([(200, {"ok": True}, None)], recorder, timeout=30.0)

    client.request("GET", "/v1/agents", timeout=5.0)
    client.request("GET", "/v1/agents")  # no override -> client default

    override_timeout = recorder.requests[0].extensions["timeout"]
    default_timeout = recorder.requests[1].extensions["timeout"]
    assert override_timeout["read"] == 5.0
    assert default_timeout["read"] == 30.0
