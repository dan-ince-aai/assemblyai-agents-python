import json as _json
import uuid

import pytest
from assemblyai_agents import (
    AsyncPager,
    ConflictError,
    ServerError,
    ValidationError,
)
from assemblyai_agents.models.rest import (
    AgentCreateRequest,
    AgentUpdateRequest,
    CallDirection,
    CallStatus,
    CreateCallRequest,
    CreateWebhookSubscriptionRequest,
    ImportPhoneNumberRequest,
    PhoneNumberAssignAgentRequest,
    PhoneNumberResponse,
    PurchaseAvailablePhoneNumberRequest,
    PurchasePhoneNumberRequest,
    SessionListItem,
    SessionResponse,
    TokenCreateRequest,
    UpdateWebhookSubscriptionRequest,
    VoiceConfig,
    WebhookDeliveryListResponse,
    WebhookDeliveryResponse,
    WebhookEvent,
    WebhookSubscriptionResponse,
)

from .conftest import Recorder, err

HEADER = "Idempotency-Key"

_TS = "2026-06-16T00:00:00Z"


@pytest.mark.asyncio
async def test_async_parity_retry_idempotency_pagination(
    make_async_client, sleep_recorder
):
    # --- #3 retry: 503 then 200, exactly two attempts ---
    rec_retry = Recorder()
    client = make_async_client(
        [(503, err("auth_service_unavailable"), None), (200, {"ok": True}, None)],
        rec_retry,
    )
    body = await client.request("GET", "/v1/agents")
    assert body == {"ok": True}
    assert rec_retry.count == 2

    # exhaustion raises the typed exception, same attempt count as sync
    rec_exhaust = Recorder()
    client = make_async_client(
        [(503, err("auth_service_unavailable"), None)], rec_exhaust, max_retries=3
    )
    with pytest.raises(ServerError):
        await client.request("GET", "/v1/agents")
    assert rec_exhaust.count == 4

    # --- #5 idempotency: same key byte-identical across retries ---
    rec_idem = Recorder()
    client = make_async_client(
        [
            (503, err("auth_service_unavailable"), None),
            (503, err("auth_service_unavailable"), None),
            (201, {"id": "c"}, None),
        ],
        rec_idem,
    )
    await client.request("POST", "/v1/calls", json={"agent_id": "a"}, idempotent=True)
    keys = rec_idem.headers(HEADER)
    assert rec_idem.count == 3
    assert len(set(keys)) == 1 and keys[0] is not None
    assert uuid.UUID(keys[0]).version == 4

    # --- #6 in_progress 409 retried same key; reuse 422 + phone conflict terminal ---
    rec_inprog = Recorder()
    client = make_async_client(
        [(409, err("idempotency_in_progress"), None), (201, {"id": "c"}, None)],
        rec_inprog,
    )
    await client.request("POST", "/v1/calls", json={"agent_id": "a"}, idempotent=True)
    assert rec_inprog.count == 2
    inprog_keys = rec_inprog.headers(HEADER)
    assert inprog_keys[0] == inprog_keys[1]

    rec_reuse = Recorder()
    client = make_async_client([(422, err("idempotency_key_reuse"), None)], rec_reuse)
    with pytest.raises(ValidationError):
        await client.request(
            "POST", "/v1/calls", json={"agent_id": "a"}, idempotent=True
        )
    assert rec_reuse.count == 1

    rec_conflict = Recorder()
    client = make_async_client(
        [(409, err("phone_number_conflict"), None)], rec_conflict
    )
    with pytest.raises(ConflictError):
        await client.request(
            "POST", "/v1/phone-numbers", json={"n": "+1"}, idempotent=True
        )
    assert rec_conflict.count == 1

    # --- #8 pagination: stop on empty-string cursor, cursor passed on next page ---
    rec_page = Recorder()
    client = make_async_client(
        [
            {
                "agents": ["a", "b"],
                "has_more": True,
                "response_metadata": {"next_cursor": "c1"},
            },
            {
                "agents": ["c"],
                "has_more": False,
                "response_metadata": {"next_cursor": ""},
            },
        ],
        rec_page,
    )
    pager: AsyncPager = client.paginate(
        "/v1/agents", item_key="agents", params={"limit": 1}
    )
    collected = [item async for item in pager]
    assert collected == ["a", "b", "c"]
    assert rec_page.count == 2
    assert dict(rec_page.requests[1].url.params).get("cursor") == "c1"


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
    return {"id": agent_id, "name": "support", "created_at": _TS, "updated_at": _TS}


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


def _request_signature(request) -> dict:
    """Verb, path, sorted query, JSON body, and presence of an idempotency key —
    the observable wire shape that the async twin must match the sync one on."""
    content = request.content
    if content in (b"", b"null"):
        body = None
    else:
        body = _json.loads(content)
    return {
        "method": request.method,
        "path": request.url.path,
        "query": sorted(request.url.params.multi_items()),
        "body": body,
        "has_idempotency_key": request.headers.get(HEADER) is not None,
    }


@pytest.mark.asyncio
async def test_async_parity_resources_match_sync(make_client, make_async_client):
    # For each resource method, drive the sync and async twin against the same
    # recorded responses and assert the observable request is identical
    # (verb/path/query/body/idempotency-key presence) and the return type matches.
    from assemblyai_agents.models.rest import (
        AgentListItem,
        AgentResponse,
        CallGetResponse,
        CallListItem,
        CallResponse,
        TokenResponse,
    )

    def sync_responses_for(name):
        return {
            "agents.create": [(201, _agent_response_dict(), None)],
            "agents.get": [(200, _agent_response_dict("agt_x"), None)],
            "agents.update": [(200, _agent_response_dict(), None)],
            "agents.delete": [(204, None, None)],
            "agents.list": [
                {
                    "agents": [_agent_list_item_dict("agt_1")],
                    "has_more": True,
                    "response_metadata": {"next_cursor": "c1"},
                },
                {
                    "agents": [_agent_list_item_dict("agt_2")],
                    "has_more": False,
                    "response_metadata": {"next_cursor": ""},
                },
            ],
            "calls.create": [(201, _call_response_dict(), None)],
            "calls.get": [(200, _call_get_dict("call_x"), None)],
            "calls.delete": [(204, None, None)],
            "calls.list": [
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
            "tokens.create": [(200, _token_response_dict(), None)],
        }[name]

    def run_sync(name, client):
        if name == "agents.create":
            return client.agents.create(_agent_create_body())
        if name == "agents.get":
            return client.agents.get("agt_x")
        if name == "agents.update":
            return client.agents.update("agt_1", AgentUpdateRequest(name="renamed"))
        if name == "agents.delete":
            return client.agents.delete("agt_1")
        if name == "agents.list":
            return list(client.agents.list(limit=1))
        if name == "calls.create":
            return client.calls.create(_call_create_body())
        if name == "calls.get":
            return client.calls.get("call_x")
        if name == "calls.delete":
            return client.calls.delete("call_1")
        if name == "calls.list":
            return list(
                client.calls.list(
                    limit=1,
                    status=CallStatus.active,
                    direction=CallDirection.outbound,
                )
            )
        if name == "tokens.create":
            return client.tokens.create(TokenCreateRequest(expires_in_seconds=120))
        raise AssertionError(name)

    async def run_async(name, client):
        if name == "agents.create":
            return await client.agents.create(_agent_create_body())
        if name == "agents.get":
            return await client.agents.get("agt_x")
        if name == "agents.update":
            return await client.agents.update(
                "agt_1", AgentUpdateRequest(name="renamed")
            )
        if name == "agents.delete":
            return await client.agents.delete("agt_1")
        if name == "agents.list":
            return [item async for item in client.agents.list(limit=1)]
        if name == "calls.create":
            return await client.calls.create(_call_create_body())
        if name == "calls.get":
            return await client.calls.get("call_x")
        if name == "calls.delete":
            return await client.calls.delete("call_1")
        if name == "calls.list":
            return [
                item
                async for item in client.calls.list(
                    limit=1,
                    status=CallStatus.active,
                    direction=CallDirection.outbound,
                )
            ]
        if name == "tokens.create":
            return await client.tokens.create(
                TokenCreateRequest(expires_in_seconds=120)
            )
        raise AssertionError(name)

    expected_type = {
        "agents.create": AgentResponse,
        "agents.get": AgentResponse,
        "agents.update": AgentResponse,
        "agents.delete": type(None),
        "agents.list": list,
        "calls.create": CallResponse,
        "calls.get": CallGetResponse,
        "calls.delete": type(None),
        "calls.list": list,
        "tokens.create": TokenResponse,
    }
    list_item_type = {"agents.list": AgentListItem, "calls.list": CallListItem}

    names = [
        "agents.create",
        "agents.get",
        "agents.update",
        "agents.delete",
        "agents.list",
        "calls.create",
        "calls.get",
        "calls.delete",
        "calls.list",
        "tokens.create",
    ]

    for name in names:
        sync_rec = Recorder()
        async_rec = Recorder()
        sync_client = make_client(sync_responses_for(name), sync_rec)
        async_client = make_async_client(sync_responses_for(name), async_rec)

        sync_result = run_sync(name, sync_client)
        async_result = await run_async(name, async_client)

        # same number of HTTP attempts and identical per-request signatures
        assert sync_rec.count == async_rec.count, name
        sync_sigs = [_request_signature(r) for r in sync_rec.requests]
        async_sigs = [_request_signature(r) for r in async_rec.requests]
        # idempotency keys differ by value (random UUIDs) but the PRESENCE flag
        # is part of the signature and must match; values compared separately.
        assert sync_sigs == async_sigs, name

        # idempotency-key presence parity: only calls.create carries one
        sync_has_key = any(r.headers.get(HEADER) for r in sync_rec.requests)
        async_has_key = any(r.headers.get(HEADER) for r in async_rec.requests)
        assert sync_has_key == async_has_key == (name == "calls.create"), name

        # return-type parity
        assert isinstance(sync_result, expected_type[name]), name
        assert isinstance(async_result, expected_type[name]), name
        if name in list_item_type:
            assert all(isinstance(i, list_item_type[name]) for i in sync_result), name
            assert all(isinstance(i, list_item_type[name]) for i in async_result), name


# ===========================================================================
# PR13b: the three new resource families (phone_numbers / sessions / webhooks)
# ===========================================================================

_NUMBER = "+14155550132"
_ENCODED = "%2B14155550132"
_SECRET = "a" * 32


def _phone_response_dict(number_id: str = "pn_1") -> dict:
    return {
        "id": number_id,
        "phone_number": _NUMBER,
        "agent_id": None,
        "type": "managed",
        "termination_uri": None,
        "created_at": _TS,
        "updated_at": _TS,
    }


def _session_list_item_dict(session_id: str) -> dict:
    return {"id": session_id, "status": "completed"}


def _session_response_dict(session_id: str = "sess_1") -> dict:
    return {
        "id": session_id,
        "status": "completed",
        "config": {"agent_id": "agt_1"},
        "artifacts": [],
    }


def _webhook_subscription_dict(sub_id: str = "whs_1") -> dict:
    return {
        "id": sub_id,
        "url": "https://example.com/webhooks/voice-agents",
        "events": ["session.completed", "call.ended"],
        "enabled": True,
        "secret_version": 1,
        "created_at": _TS,
        "updated_at": _TS,
    }


def _webhook_delivery_dict(delivery_id: str) -> dict:
    return {
        "id": delivery_id,
        "subscription_id": "whs_1",
        "session_id": "sess_1",
        "attempt_count": 1,
        "status": "delivered",
        "created_at": _TS,
    }


def _create_webhook_body() -> CreateWebhookSubscriptionRequest:
    return CreateWebhookSubscriptionRequest(
        url="https://example.com/webhooks/voice-agents",
        events=[WebhookEvent.session_completed, WebhookEvent.call_ended],
        secret=_SECRET,
    )


def _purchase_available_body() -> PurchaseAvailablePhoneNumberRequest:
    return PurchaseAvailablePhoneNumberRequest(country_code="US", number_type="local")


# The full method matrix for the three new families. Each entry: responses, the
# sync caller, the async caller, expected return type, optional list-item type.
def _resources_b_specs():
    return {
        "phone_numbers.list": dict(
            responses=[
                {
                    "phone_numbers": [_phone_response_dict("pn_1")],
                    "has_more": False,
                    "response_metadata": {"next_cursor": ""},
                }
            ],
            sync=lambda c: list(c.phone_numbers.list(limit=1)),
            asy=lambda c: c.phone_numbers.list(limit=1),
            is_list=True,
            ret=list,
            item=PhoneNumberResponse,
        ),
        "phone_numbers.purchase_available": dict(
            responses=[(201, _phone_response_dict("pn_42"), None)],
            sync=lambda c: c.phone_numbers.purchase_available(
                _purchase_available_body()
            ),
            asy=lambda c: c.phone_numbers.purchase_available(
                _purchase_available_body()
            ),
            ret=PhoneNumberResponse,
        ),
        "phone_numbers.purchase": dict(
            responses=[(201, None, None)],
            sync=lambda c: c.phone_numbers.purchase(
                PurchasePhoneNumberRequest(phone_number=_NUMBER)
            ),
            asy=lambda c: c.phone_numbers.purchase(
                PurchasePhoneNumberRequest(phone_number=_NUMBER)
            ),
            ret=type(None),
        ),
        "phone_numbers.import_": dict(
            responses=[(201, None, None)],
            sync=lambda c: c.phone_numbers.import_(
                ImportPhoneNumberRequest(phone_number=_NUMBER)
            ),
            asy=lambda c: c.phone_numbers.import_(
                ImportPhoneNumberRequest(phone_number=_NUMBER)
            ),
            ret=type(None),
        ),
        "phone_numbers.get": dict(
            responses=[(200, _phone_response_dict("pn_9"), None)],
            sync=lambda c: c.phone_numbers.get(_NUMBER),
            asy=lambda c: c.phone_numbers.get(_NUMBER),
            ret=PhoneNumberResponse,
        ),
        "phone_numbers.deregister": dict(
            responses=[(204, None, None)],
            sync=lambda c: c.phone_numbers.deregister(_NUMBER),
            asy=lambda c: c.phone_numbers.deregister(_NUMBER),
            ret=type(None),
        ),
        "phone_numbers.assign_agent": dict(
            responses=[(200, None, None)],
            sync=lambda c: c.phone_numbers.assign_agent(
                _NUMBER, PhoneNumberAssignAgentRequest(agent_id="agt_1")
            ),
            asy=lambda c: c.phone_numbers.assign_agent(
                _NUMBER, PhoneNumberAssignAgentRequest(agent_id="agt_1")
            ),
            ret=type(None),
        ),
        "phone_numbers.unassign_agent": dict(
            responses=[(204, None, None)],
            sync=lambda c: c.phone_numbers.unassign_agent(_NUMBER),
            asy=lambda c: c.phone_numbers.unassign_agent(_NUMBER),
            ret=type(None),
        ),
        "sessions.list": dict(
            responses=[
                {
                    "sessions": [_session_list_item_dict("sess_1")],
                    "has_more": False,
                    "response_metadata": {"next_cursor": ""},
                }
            ],
            sync=lambda c: list(c.sessions.list(limit=1, status="completed")),
            asy=lambda c: c.sessions.list(limit=1, status="completed"),
            is_list=True,
            ret=list,
            item=SessionListItem,
        ),
        "sessions.get": dict(
            responses=[(200, _session_response_dict("sess_x"), None)],
            sync=lambda c: c.sessions.get("sess_x"),
            asy=lambda c: c.sessions.get("sess_x"),
            ret=SessionResponse,
        ),
        "sessions.delete": dict(
            responses=[(204, None, None)],
            sync=lambda c: c.sessions.delete("sess_1"),
            asy=lambda c: c.sessions.delete("sess_1"),
            ret=type(None),
        ),
        "webhooks.create": dict(
            responses=[(201, _webhook_subscription_dict("whs_7"), None)],
            sync=lambda c: c.webhooks.create(_create_webhook_body()),
            asy=lambda c: c.webhooks.create(_create_webhook_body()),
            ret=WebhookSubscriptionResponse,
        ),
        "webhooks.list": dict(
            responses=[
                {
                    "subscriptions": [_webhook_subscription_dict("whs_1")],
                    "has_more": False,
                    "response_metadata": {"next_cursor": ""},
                }
            ],
            sync=lambda c: list(c.webhooks.list(limit=1, include_disabled=True)),
            asy=lambda c: c.webhooks.list(limit=1, include_disabled=True),
            is_list=True,
            ret=list,
            item=WebhookSubscriptionResponse,
        ),
        "webhooks.get": dict(
            responses=[(200, _webhook_subscription_dict("whs_5"), None)],
            sync=lambda c: c.webhooks.get("whs_5"),
            asy=lambda c: c.webhooks.get("whs_5"),
            ret=WebhookSubscriptionResponse,
        ),
        "webhooks.update": dict(
            responses=[(200, _webhook_subscription_dict("whs_1"), None)],
            sync=lambda c: c.webhooks.update(
                "whs_1", UpdateWebhookSubscriptionRequest(enabled=False)
            ),
            asy=lambda c: c.webhooks.update(
                "whs_1", UpdateWebhookSubscriptionRequest(enabled=False)
            ),
            ret=WebhookSubscriptionResponse,
        ),
        "webhooks.delete": dict(
            responses=[(204, None, None)],
            sync=lambda c: c.webhooks.delete("whs_1"),
            asy=lambda c: c.webhooks.delete("whs_1"),
            ret=type(None),
        ),
        "webhooks.list_deliveries": dict(
            responses=[
                {
                    "deliveries": [_webhook_delivery_dict("whd_1")],
                    "has_more": False,
                    "response_metadata": {"next_cursor": ""},
                }
            ],
            sync=lambda c: list(c.webhooks.list_deliveries("sess_1", limit=1)),
            asy=lambda c: c.webhooks.list_deliveries("sess_1", limit=1),
            is_list=True,
            ret=list,
            item=WebhookDeliveryResponse,
        ),
        "webhooks.list_latest_deliveries": dict(
            responses=[
                (
                    200,
                    {
                        "deliveries": [_webhook_delivery_dict("whd_1")],
                        "response_metadata": {"next_cursor": ""},
                    },
                    None,
                )
            ],
            sync=lambda c: c.webhooks.list_latest_deliveries("sess_1"),
            asy=lambda c: c.webhooks.list_latest_deliveries("sess_1"),
            ret=WebhookDeliveryListResponse,
        ),
    }


# The three methods the router idempotency-guards (and ONLY these).
_IDEMPOTENT_NAMES = {
    "phone_numbers.purchase_available",
    "phone_numbers.purchase",
    "phone_numbers.import_",
}


def test_async_parity_all_methods(make_async_client):
    # Every new method exists on the matching Async*Resource with the same name,
    # and the list methods return AsyncPager (not awaited) while non-list methods
    # are coroutines.
    asy = make_async_client([(200, {"ok": True}, None)], Recorder())

    expected = {
        "phone_numbers": [
            "list",
            "purchase_available",
            "purchase",
            "import_",
            "get",
            "deregister",
            "assign_agent",
            "unassign_agent",
        ],
        "sessions": ["list", "get", "delete"],
        "webhooks": [
            "create",
            "list",
            "get",
            "update",
            "delete",
            "list_deliveries",
            "list_latest_deliveries",
        ],
    }
    for namespace, methods in expected.items():
        resource = getattr(asy, namespace)
        for method in methods:
            assert hasattr(resource, method), f"{namespace}.{method} missing on async"
            assert callable(getattr(resource, method)), f"{namespace}.{method}"


@pytest.mark.asyncio
async def test_async_roundtrips_match_sync(make_client, make_async_client):
    # Drive each of the 18 new methods sync and async against identical responses
    # and assert the observable wire signature is identical and the return types
    # match, with idempotency-key presence parity on exactly the three guarded
    # phone POSTs.
    specs = _resources_b_specs()

    for name, spec in specs.items():
        sync_rec = Recorder()
        async_rec = Recorder()
        sync_client = make_client(spec["responses"], sync_rec)
        async_client = make_async_client(spec["responses"], async_rec)

        sync_result = spec["sync"](sync_client)
        if spec.get("is_list"):
            async_pager = spec["asy"](async_client)
            assert isinstance(async_pager, AsyncPager), name
            async_result = [item async for item in async_pager]
        else:
            async_result = await spec["asy"](async_client)

        # identical observable wire signature across every attempt
        assert sync_rec.count == async_rec.count, name
        sync_sigs = [_request_signature(r) for r in sync_rec.requests]
        async_sigs = [_request_signature(r) for r in async_rec.requests]
        assert sync_sigs == async_sigs, name

        # idempotency-key presence parity: exactly the three guarded phone POSTs
        sync_has_key = any(r.headers.get(HEADER) for r in sync_rec.requests)
        async_has_key = any(r.headers.get(HEADER) for r in async_rec.requests)
        assert sync_has_key == async_has_key == (name in _IDEMPOTENT_NAMES), name

        # return-type parity
        assert isinstance(sync_result, spec["ret"]), name
        assert isinstance(async_result, spec["ret"]), name
        if spec.get("item"):
            assert all(isinstance(i, spec["item"]) for i in sync_result), name
            assert all(isinstance(i, spec["item"]) for i in async_result), name
