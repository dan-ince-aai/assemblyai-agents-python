import uuid

from assemblyai_agents import SyncPager
from assemblyai_agents.models.rest import (
    AgentCreateRequest,
    AgentListItem,
    AgentResponse,
    AgentUpdateRequest,
    CallDirection,
    CallGetResponse,
    CallListItem,
    CallResponse,
    CallStatus,
    CreateCallRequest,
    TokenCreateRequest,
    TokenResponse,
    VoiceConfig,
)
from assemblyai_agents.resources.agents import AgentsResource, AsyncAgentsResource
from assemblyai_agents.resources.calls import AsyncCallsResource, CallsResource
from assemblyai_agents.resources.tokens import AsyncTokensResource, TokensResource

from .conftest import Recorder, err

HEADER = "Idempotency-Key"

_TS = "2026-06-16T00:00:00Z"


def _agent_create_body() -> AgentCreateRequest:
    return AgentCreateRequest(
        name="support",
        system_prompt="You are a helpful support agent.",
        voice=VoiceConfig(voice_id="ivy"),
    )


def _agent_response_dict(agent_id: str = "agt_1") -> dict:
    return {
        "id": agent_id,
        "name": "support",
        "system_prompt": "You are a helpful support agent.",
        "voice": {"voice_id": "ivy"},
        "input": {},
        "output": {},
        "created_at": _TS,
        "updated_at": _TS,
    }


def _agent_list_item_dict(agent_id: str) -> dict:
    return {
        "id": agent_id,
        "name": "support",
        "created_at": _TS,
        "updated_at": _TS,
    }


def _call_create_body() -> CreateCallRequest:
    return CreateCallRequest(from_number="+14155550132", to_number="+12125550148")


def _call_response_dict(call_id: str = "call_1") -> dict:
    return {"id": call_id, "status": "dialing"}


def _call_get_dict(call_id: str = "call_1") -> dict:
    return {
        "id": call_id,
        "status": "active",
        "direction": "outbound",
        "agent_id": "agt_1",
        "from_number": "+14155550132",
        "to_number": "+12125550148",
        "session_id": "sess_1",
        "created_at": _TS,
        "updated_at": _TS,
    }


def _call_list_item_dict(call_id: str) -> dict:
    return {
        "id": call_id,
        "status": "ended",
        "direction": "inbound",
        "agent_id": "agt_1",
        "from_number": "+14155550132",
        "to_number": "+12125550148",
        "session_id": "sess_1",
        "created_at": _TS,
        "updated_at": _TS,
    }


def _token_response_dict() -> dict:
    return {"token": "tkn_live_abc", "expires_at": _TS}


def _body_json(request) -> dict:
    import json as _json

    return _json.loads(request.content)


# --- fail-at-base anchors ---------------------------------------------------


def test_resources_absent_at_base(make_client, recorder: Recorder):
    # Exercises the new layer end to end: build a resource directly over the
    # mock transport and route a create through it. At base this import (and the
    # resources subpackage) does not exist, so the module fails to import and
    # the test errors; at HEAD it posts and parses an AgentResponse.
    client = make_client([(201, _agent_response_dict(), None)], recorder)
    resource = AgentsResource(client)
    result = resource.create(_agent_create_body())
    assert isinstance(result, AgentResponse)
    assert recorder.requests[0].method == "POST"
    assert recorder.requests[0].url.path == "/v1/agents"


def test_eager_resource_attach_correct_classes(
    make_client, make_async_client, recorder: Recorder
):
    sync_client = make_client([(200, {"ok": True}, None)], recorder)
    assert isinstance(sync_client.agents, AgentsResource)
    assert isinstance(sync_client.calls, CallsResource)
    assert isinstance(sync_client.tokens, TokensResource)

    async_client = make_async_client([(200, {"ok": True}, None)], Recorder())
    assert isinstance(async_client.agents, AsyncAgentsResource)
    assert isinstance(async_client.calls, AsyncCallsResource)
    assert isinstance(async_client.tokens, AsyncTokensResource)
    # sync client must NOT carry the async classes (no cross-wiring)
    assert not isinstance(sync_client.agents, AsyncAgentsResource)


# --- agents -----------------------------------------------------------------


def test_agents_create_posts_request_returns_agentresponse(
    make_client, recorder: Recorder
):
    client = make_client([(201, _agent_response_dict(), None)], recorder)

    result = client.agents.create(_agent_create_body())

    assert isinstance(result, AgentResponse)
    assert result.id == "agt_1"
    req = recorder.requests[0]
    assert req.method == "POST"
    assert req.url.path == "/v1/agents"
    body = _body_json(req)
    assert body["name"] == "support"
    assert body["system_prompt"] == "You are a helpful support agent."
    assert body["voice"] == {"voice_id": "ivy"}
    # unset Optionals dropped by exclude_none
    assert "greeting" not in body
    assert "tools" not in body
    assert recorder.header(HEADER) is None


def test_agents_list_pages_typed_and_stops_on_empty_cursor(
    make_client, recorder: Recorder
):
    client = make_client(
        [
            {
                "agents": [
                    _agent_list_item_dict("agt_1"),
                    _agent_list_item_dict("agt_2"),
                ],
                "has_more": True,
                "response_metadata": {"next_cursor": "c1"},
            },
            {
                "agents": [_agent_list_item_dict("agt_3")],
                "has_more": False,
                "response_metadata": {"next_cursor": ""},
            },
        ],
        recorder,
    )

    pager = client.agents.list(limit=2)
    assert isinstance(pager, SyncPager)
    items = list(pager)

    assert [type(i) for i in items] == [AgentListItem, AgentListItem, AgentListItem]
    assert [i.id for i in items] == ["agt_1", "agt_2", "agt_3"]
    assert recorder.count == 2
    assert all(r.method == "GET" for r in recorder.requests)
    assert recorder.requests[0].url.path == "/v1/agents"
    # cursor threaded onto the second page only
    assert "cursor" not in dict(recorder.requests[0].url.params)
    assert dict(recorder.requests[1].url.params).get("cursor") == "c1"


def test_agents_get_substitutes_id_returns_agentresponse(
    make_client, recorder: Recorder
):
    client = make_client([(200, _agent_response_dict("agt_xyz"), None)], recorder)

    result = client.agents.get("agt_xyz")

    assert isinstance(result, AgentResponse)
    assert result.id == "agt_xyz"
    req = recorder.requests[0]
    assert req.method == "GET"
    assert req.url.path == "/v1/agents/agt_xyz"


def test_agents_update_omits_tools_and_no_masked_sentinel(
    make_client, recorder: Recorder
):
    client = make_client([(200, _agent_response_dict("agt_1"), None)], recorder)

    # tools deliberately unset on the update request
    result = client.agents.update("agt_1", AgentUpdateRequest(name="renamed"))

    assert isinstance(result, AgentResponse)
    req = recorder.requests[0]
    assert req.method == "PUT"
    assert req.url.path == "/v1/agents/agt_1"
    body = _body_json(req)
    assert body == {"name": "renamed"}
    assert "tools" not in body
    assert "***" not in req.content.decode("utf-8")
    assert recorder.header(HEADER) is None


def test_agents_delete_returns_none_on_204(make_client, recorder: Recorder):
    client = make_client([(204, None, None)], recorder)

    result = client.agents.delete("agt_1")

    assert result is None
    req = recorder.requests[0]
    assert req.method == "DELETE"
    assert req.url.path == "/v1/agents/agt_1"


# --- calls ------------------------------------------------------------------


def test_calls_create_sends_idempotency_key_stable_across_retries(
    make_client, recorder: Recorder, sleep_recorder
):
    client = make_client(
        [
            (503, err("auth_service_unavailable"), None),
            (409, err("idempotency_in_progress"), None),
            (201, _call_response_dict(), None),
        ],
        recorder,
    )

    result = client.calls.create(_call_create_body())

    assert isinstance(result, CallResponse)
    assert recorder.count == 3
    keys = recorder.headers(HEADER)
    assert all(k is not None for k in keys), keys
    assert len(set(keys)) == 1, f"key changed across retries: {keys}"
    assert uuid.UUID(keys[0]).version == 4
    req = recorder.requests[0]
    assert req.method == "POST"
    assert req.url.path == "/v1/calls"
    body = _body_json(req)
    assert body == {"from_number": "+14155550132", "to_number": "+12125550148"}


def test_calls_create_returns_callresponse(make_client, recorder: Recorder):
    client = make_client([(201, _call_response_dict("call_42"), None)], recorder)

    result = client.calls.create(_call_create_body())

    # The minimal create shape, NOT CallGetResponse.
    assert isinstance(result, CallResponse)
    assert not isinstance(result, CallGetResponse)
    assert result.id == "call_42"
    assert result.status == CallStatus.dialing


def test_calls_list_pages_typed_with_status_direction_filters(
    make_client, recorder: Recorder
):
    client = make_client(
        [
            {
                "calls": [_call_list_item_dict("call_1")],
                "has_more": True,
                "response_metadata": {"next_cursor": "c1"},
            },
            {
                "calls": [_call_list_item_dict("call_2")],
                "has_more": False,
                "response_metadata": {"next_cursor": ""},
            },
        ],
        recorder,
    )

    pager = client.calls.list(
        limit=1, status=CallStatus.active, direction=CallDirection.outbound
    )
    assert isinstance(pager, SyncPager)
    items = list(pager)

    assert [type(i) for i in items] == [CallListItem, CallListItem]
    assert [i.id for i in items] == ["call_1", "call_2"]
    assert recorder.count == 2
    first_q = dict(recorder.requests[0].url.params)
    assert recorder.requests[0].url.path == "/v1/calls"
    # enums serialized to their .value strings in the query
    assert first_q.get("status") == "active"
    assert first_q.get("direction") == "outbound"
    assert dict(recorder.requests[1].url.params).get("cursor") == "c1"
    assert all(r.headers.get(HEADER) is None for r in recorder.requests)


def test_calls_get_returns_callgetresponse(make_client, recorder: Recorder):
    client = make_client([(200, _call_get_dict("call_9"), None)], recorder)

    result = client.calls.get("call_9")

    # The fuller get shape, distinct from CallResponse.
    assert isinstance(result, CallGetResponse)
    assert not isinstance(result, CallResponse)
    assert result.id == "call_9"
    assert result.session_id == "sess_1"
    req = recorder.requests[0]
    assert req.method == "GET"
    assert req.url.path == "/v1/calls/call_9"


def test_calls_delete_returns_none_on_204(make_client, recorder: Recorder):
    client = make_client([(204, None, None)], recorder)

    result = client.calls.delete("call_1")

    assert result is None
    req = recorder.requests[0]
    assert req.method == "DELETE"
    assert req.url.path == "/v1/calls/call_1"


# --- tokens -----------------------------------------------------------------


def test_tokens_create_minimal_body_returns_tokenresponse(
    make_client, recorder: Recorder
):
    client = make_client([(200, _token_response_dict(), None)], recorder)

    # explicit body: only expires_in_seconds on the wire
    result = client.tokens.create(TokenCreateRequest(expires_in_seconds=120))

    assert isinstance(result, TokenResponse)
    assert result.token == "tkn_live_abc"
    req = recorder.requests[0]
    assert req.method == "POST"
    assert req.url.path == "/v1/tokens"
    body = _body_json(req)
    assert set(body.keys()) <= {"expires_in_seconds"}
    assert body["expires_in_seconds"] == 120
    assert recorder.header(HEADER) is None

    # body=None path: no JSON body sent, still returns 200 TokenResponse
    rec_none = Recorder()
    client_none = make_client([(200, _token_response_dict(), None)], rec_none)
    result_none = client_none.tokens.create()
    assert isinstance(result_none, TokenResponse)
    none_body = rec_none.requests[0].content
    assert none_body in (b"", b"null"), none_body
    assert rec_none.header(HEADER) is None


def test_idempotency_selectivity_only_calls_create(make_client):
    # agents.create / agents.update / tokens.create send NO Idempotency-Key;
    # only calls.create does (asserted in its own dedicated test).
    rec_agent_create = Recorder()
    c1 = make_client([(201, _agent_response_dict(), None)], rec_agent_create)
    c1.agents.create(_agent_create_body())
    assert rec_agent_create.header(HEADER) is None

    rec_agent_update = Recorder()
    c2 = make_client([(200, _agent_response_dict(), None)], rec_agent_update)
    c2.agents.update("agt_1", AgentUpdateRequest(name="x"))
    assert rec_agent_update.header(HEADER) is None

    rec_token = Recorder()
    c3 = make_client([(200, _token_response_dict(), None)], rec_token)
    c3.tokens.create(TokenCreateRequest(expires_in_seconds=60))
    assert rec_token.header(HEADER) is None

    # contrast: calls.create DOES carry the key
    rec_call = Recorder()
    c4 = make_client([(201, _call_response_dict(), None)], rec_call)
    c4.calls.create(_call_create_body())
    assert rec_call.header(HEADER) is not None
    assert uuid.UUID(rec_call.header(HEADER)).version == 4
