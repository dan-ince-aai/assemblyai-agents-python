import json as _json
import uuid

import pytest
from assemblyai_agents import ConfigurationError, SyncPager, VoiceAgent, tool
from assemblyai_agents.models.rest import (
    CreateWebhookSubscriptionRequest,
    ImportPhoneNumberRequest,
    PhoneNumberAssignAgentRequest,
    PhoneNumberResponse,
    PlaintextHttpToolConfig,
    PurchaseAvailablePhoneNumberRequest,
    PurchasePhoneNumberRequest,
    SessionListItem,
    SessionResponse,
    UpdateWebhookSubscriptionRequest,
    WebhookDeliveryListResponse,
    WebhookDeliveryResponse,
    WebhookEvent,
    WebhookSubscriptionResponse,
)
from assemblyai_agents.resources.phone_numbers import (
    AsyncPhoneNumbersResource,
    PhoneNumbersResource,
)
from assemblyai_agents.resources.sessions import (
    AsyncSessionsResource,
    SessionsResource,
)
from assemblyai_agents.resources.webhooks import (
    AsyncWebhooksResource,
    WebhooksResource,
)

from .conftest import Recorder

HEADER = "Idempotency-Key"

_TS = "2026-06-16T00:00:00Z"
_NUMBER = "+14155550132"
_ENCODED = "%2B14155550132"
_SECRET = "a" * 32


def _body_json(request) -> dict:
    return _json.loads(request.content)


# --- response/body builders -------------------------------------------------


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


# ===========================================================================
# phone_numbers — per-method round-trips (AC-1)
# ===========================================================================


def test_phone_numbers_list_roundtrip(make_client, recorder: Recorder):
    client = make_client(
        [
            {
                "phone_numbers": [_phone_response_dict("pn_1")],
                "has_more": False,
                "response_metadata": {"next_cursor": ""},
            }
        ],
        recorder,
    )

    pager = client.phone_numbers.list(limit=10)
    assert isinstance(pager, SyncPager)
    items = list(pager)

    assert [type(i) for i in items] == [PhoneNumberResponse]
    assert items[0].phone_number == _NUMBER
    req = recorder.requests[0]
    assert req.method == "GET"
    assert req.url.path == "/v1/phone-numbers"
    assert dict(req.url.params).get("limit") == "10"
    assert recorder.header(HEADER) is None


def test_phone_numbers_purchase_available_roundtrip(make_client, recorder: Recorder):
    client = make_client([(201, _phone_response_dict("pn_42"), None)], recorder)

    result = client.phone_numbers.purchase_available(_purchase_available_body())

    assert isinstance(result, PhoneNumberResponse)
    assert result.id == "pn_42"
    req = recorder.requests[0]
    assert req.method == "POST"
    assert req.url.path == "/v1/phone-numbers"
    body = _body_json(req)
    assert body["country_code"] == "US"
    assert body["number_type"] == "local"
    # exclude_none drops the unset optionals
    assert "area_code" not in body
    assert "agent_id" not in body


def test_phone_numbers_purchase_returns_none(make_client, recorder: Recorder):
    # empty-body 201 -> None via request_raw (no model_validate)
    client = make_client([(201, None, None)], recorder)

    result = client.phone_numbers.purchase(
        PurchasePhoneNumberRequest(phone_number=_NUMBER)
    )

    assert result is None
    req = recorder.requests[0]
    assert req.method == "POST"
    assert req.url.path == "/v1/phone-numbers/purchase"
    body = _body_json(req)
    assert body == {"phone_number": _NUMBER}


def test_phone_numbers_import_returns_none(make_client, recorder: Recorder):
    client = make_client([(201, None, None)], recorder)

    result = client.phone_numbers.import_(
        ImportPhoneNumberRequest(phone_number=_NUMBER)
    )

    assert result is None
    req = recorder.requests[0]
    assert req.method == "POST"
    assert req.url.path == "/v1/phone-numbers/import"
    body = _body_json(req)
    assert body == {"phone_number": _NUMBER}


def test_phone_numbers_get_roundtrip(make_client, recorder: Recorder):
    client = make_client([(200, _phone_response_dict("pn_9"), None)], recorder)

    result = client.phone_numbers.get(_NUMBER)

    assert isinstance(result, PhoneNumberResponse)
    assert result.id == "pn_9"
    req = recorder.requests[0]
    assert req.method == "GET"
    # leading + must be percent-encoded (AC-4); covered in depth below
    assert req.url.raw_path.decode().startswith(f"/v1/phone-numbers/{_ENCODED}")
    assert recorder.header(HEADER) is None


def test_phone_numbers_deregister_returns_none(make_client, recorder: Recorder):
    client = make_client([(204, None, None)], recorder)

    result = client.phone_numbers.deregister(_NUMBER)

    assert result is None
    req = recorder.requests[0]
    assert req.method == "DELETE"
    assert req.url.raw_path.decode() == f"/v1/phone-numbers/{_ENCODED}"


def test_phone_numbers_assign_agent_returns_none(make_client, recorder: Recorder):
    # 200 with empty JSON schema -> None via request_raw, body still sent
    client = make_client([(200, None, None)], recorder)

    result = client.phone_numbers.assign_agent(
        _NUMBER, PhoneNumberAssignAgentRequest(agent_id="agt_1")
    )

    assert result is None
    req = recorder.requests[0]
    assert req.method == "PUT"
    assert req.url.raw_path.decode() == f"/v1/phone-numbers/{_ENCODED}/agent"
    body = _body_json(req)
    assert body == {"agent_id": "agt_1"}
    # assign_agent is NOT idempotency-guarded by the router
    assert recorder.header(HEADER) is None


def test_phone_numbers_assign_agent_allows_a_local_agent_with_only_http_tools(
    make_client, recorder: Recorder
):
    @tool(http=PlaintextHttpToolConfig(url="https://example.com/orders"))
    async def lookup_order(order_id: str) -> dict:
        """Look up one of the caller's orders by its ID."""
        return {}

    agent = VoiceAgent(
        name="Pizza Line",
        voice="ivy",
        system_prompt="Take orders.",
        tools=[lookup_order],
    )
    client = make_client([(200, None, None)], recorder)

    assert (
        client.phone_numbers.assign_agent(
            _NUMBER, PhoneNumberAssignAgentRequest(agent_id="agt_1"), agent=agent
        )
        is None
    )


def test_phone_numbers_assign_agent_skips_the_check_for_a_bare_id(
    make_client, recorder: Recorder
):
    # Only what is already in hand is checked. With an id and no declaration the
    # SDK cannot know what tools the stored agent holds, and it does not fetch.
    client = make_client([(200, None, None)], recorder)

    assert (
        client.phone_numbers.assign_agent(
            _NUMBER, PhoneNumberAssignAgentRequest(agent_id="agt_1")
        )
        is None
    )
    assert recorder.count == 1


def test_phone_numbers_unassign_agent_returns_none(make_client, recorder: Recorder):
    client = make_client([(204, None, None)], recorder)

    result = client.phone_numbers.unassign_agent(_NUMBER)

    assert result is None
    req = recorder.requests[0]
    assert req.method == "DELETE"
    assert req.url.raw_path.decode() == f"/v1/phone-numbers/{_ENCODED}/agent"


# ===========================================================================
# sessions — per-method round-trips (AC-2)
# ===========================================================================


def test_sessions_list_roundtrip(make_client, recorder: Recorder):
    client = make_client(
        [
            {
                "sessions": [
                    _session_list_item_dict("sess_1"),
                    _session_list_item_dict("sess_2"),
                ],
                "has_more": False,
                "response_metadata": {"next_cursor": ""},
            }
        ],
        recorder,
    )

    pager = client.sessions.list(limit=5)
    assert isinstance(pager, SyncPager)
    items = list(pager)

    assert [type(i) for i in items] == [SessionListItem, SessionListItem]
    assert [i.id for i in items] == ["sess_1", "sess_2"]
    req = recorder.requests[0]
    assert req.method == "GET"
    assert req.url.path == "/v1/sessions"
    assert recorder.header(HEADER) is None


def test_sessions_get_returns_full_session_response(make_client, recorder: Recorder):
    client = make_client([(200, _session_response_dict("sess_x"), None)], recorder)

    result = client.sessions.get("sess_x")

    # full model, NOT the slim list item
    assert isinstance(result, SessionResponse)
    assert not isinstance(result, SessionListItem)
    assert result.id == "sess_x"
    # config/artifacts only exist on the full SessionResponse
    assert result.config == {"agent_id": "agt_1"}
    assert result.artifacts == []
    # the slim list item has no config field at all
    assert "config" not in SessionListItem.model_fields
    req = recorder.requests[0]
    assert req.method == "GET"
    assert req.url.path == "/v1/sessions/sess_x"


def test_sessions_delete_returns_none(make_client, recorder: Recorder):
    client = make_client([(204, None, None)], recorder)

    result = client.sessions.delete("sess_1")

    assert result is None
    req = recorder.requests[0]
    assert req.method == "DELETE"
    assert req.url.path == "/v1/sessions/sess_1"
    assert recorder.header(HEADER) is None


# ===========================================================================
# webhooks — per-method round-trips (AC-3)
# ===========================================================================


def test_webhooks_create_roundtrip(make_client, recorder: Recorder):
    client = make_client([(201, _webhook_subscription_dict("whs_7"), None)], recorder)

    result = client.webhooks.create(_create_webhook_body())

    assert isinstance(result, WebhookSubscriptionResponse)
    assert result.id == "whs_7"
    # response model has no secret, only secret_version
    assert not hasattr(result, "secret")
    assert result.secret_version == 1
    req = recorder.requests[0]
    assert req.method == "POST"
    assert req.url.path == "/v1/webhook-subscriptions"
    body = _body_json(req)
    assert body["url"] == "https://example.com/webhooks/voice-agents"
    assert body["events"] == ["session.completed", "call.ended"]
    # create is NOT idempotency-guarded
    assert recorder.header(HEADER) is None


def test_webhooks_list_roundtrip(make_client, recorder: Recorder):
    client = make_client(
        [
            {
                "subscriptions": [_webhook_subscription_dict("whs_1")],
                "has_more": False,
                "response_metadata": {"next_cursor": ""},
            }
        ],
        recorder,
    )

    pager = client.webhooks.list(limit=10)
    assert isinstance(pager, SyncPager)
    items = list(pager)

    assert [type(i) for i in items] == [WebhookSubscriptionResponse]
    assert items[0].id == "whs_1"
    req = recorder.requests[0]
    assert req.method == "GET"
    assert req.url.path == "/v1/webhook-subscriptions"


def test_webhooks_get_roundtrip(make_client, recorder: Recorder):
    client = make_client([(200, _webhook_subscription_dict("whs_5"), None)], recorder)

    result = client.webhooks.get("whs_5")

    assert isinstance(result, WebhookSubscriptionResponse)
    assert result.id == "whs_5"
    req = recorder.requests[0]
    assert req.method == "GET"
    assert req.url.path == "/v1/webhook-subscriptions/whs_5"


def test_webhooks_update_is_patch(make_client, recorder: Recorder):
    client = make_client([(200, _webhook_subscription_dict("whs_1"), None)], recorder)

    result = client.webhooks.update(
        "whs_1", UpdateWebhookSubscriptionRequest(enabled=False)
    )

    assert isinstance(result, WebhookSubscriptionResponse)
    req = recorder.requests[0]
    # the whole point: PATCH, not PUT, not POST
    assert req.method == "PATCH"
    assert req.url.path == "/v1/webhook-subscriptions/whs_1"
    body = _body_json(req)
    # partial update: only the set field is on the wire
    assert body == {"enabled": False}
    assert "url" not in body
    assert "events" not in body
    assert recorder.header(HEADER) is None


def test_webhooks_delete_returns_none(make_client, recorder: Recorder):
    client = make_client([(204, None, None)], recorder)

    result = client.webhooks.delete("whs_1")

    assert result is None
    req = recorder.requests[0]
    assert req.method == "DELETE"
    assert req.url.path == "/v1/webhook-subscriptions/whs_1"
    assert recorder.header(HEADER) is None


def test_webhooks_list_deliveries_roundtrip(make_client, recorder: Recorder):
    client = make_client(
        [
            {
                "deliveries": [_webhook_delivery_dict("whd_1")],
                "has_more": False,
                "response_metadata": {"next_cursor": ""},
            }
        ],
        recorder,
    )

    pager = client.webhooks.list_deliveries("sess_1", limit=10)
    assert isinstance(pager, SyncPager)
    items = list(pager)

    assert [type(i) for i in items] == [WebhookDeliveryResponse]
    assert items[0].id == "whd_1"
    req = recorder.requests[0]
    assert req.method == "GET"
    assert req.url.path == "/v1/sessions/sess_1/webhook-deliveries"


def test_webhooks_list_latest_returns_parsed_envelope_not_pager(
    make_client, recorder: Recorder
):
    client = make_client(
        [
            (
                200,
                {
                    "deliveries": [_webhook_delivery_dict("whd_1")],
                    "has_more": False,
                    "response_metadata": {"next_cursor": ""},
                },
                None,
            )
        ],
        recorder,
    )

    result = client.webhooks.list_latest_deliveries("sess_1")

    # parsed envelope, NOT a pager
    assert isinstance(result, WebhookDeliveryListResponse)
    assert not isinstance(result, SyncPager)
    assert result.deliveries is not None
    assert [type(d) for d in result.deliveries] == [WebhookDeliveryResponse]
    req = recorder.requests[0]
    assert req.method == "GET"
    assert req.url.path == "/v1/sessions/sess_1/webhook-deliveries/latest"


# ===========================================================================
# webhook secret on the wire (AC-13) + never logged (Risk 2)
# ===========================================================================


def test_webhook_secret_real_value_on_wire_not_masked(make_client, recorder: Recorder):
    client = make_client([(201, _webhook_subscription_dict(), None)], recorder)

    client.webhooks.create(_create_webhook_body())

    body = _json.loads(recorder.requests[0].content)
    # the REAL secret must reach the wire, not pydantic's mask
    assert body["secret"] == _SECRET
    assert body["secret"] != "**********"
    # and it must be a plain string, not a nested SecretStr repr
    assert isinstance(body["secret"], str)


def test_webhook_secret_never_logged_or_repr(make_client, recorder: Recorder, caplog):
    import logging

    caplog.set_level(logging.DEBUG)
    client = make_client([(201, _webhook_subscription_dict(), None)], recorder)
    body = _create_webhook_body()

    client.webhooks.create(body)

    # the SecretStr field itself still masks on repr/str (not mutated)
    assert _SECRET not in repr(body)
    assert _SECRET not in str(body.secret)
    assert repr(body.secret) == "SecretStr('**********')"
    # client repr never carries a secret (it only prints base_url)
    assert _SECRET not in repr(client)
    # nothing the SDK logged contains the literal secret
    for record in caplog.records:
        assert _SECRET not in record.getMessage()


# ===========================================================================
# path encoding (AC-4 / AC-5)
# ===========================================================================


def test_phone_number_path_is_percent_encoded(make_client):
    # every {number} path op must percent-encode the leading + (and never emit a
    # literal + or a space). Drive all four number-keyed ops.
    for op in ("get", "deregister", "assign_agent", "unassign_agent"):
        rec = Recorder()
        if op == "get":
            client = make_client([(200, _phone_response_dict(), None)], rec)
            client.phone_numbers.get(_NUMBER)
        elif op == "deregister":
            client = make_client([(204, None, None)], rec)
            client.phone_numbers.deregister(_NUMBER)
        elif op == "assign_agent":
            client = make_client([(200, None, None)], rec)
            client.phone_numbers.assign_agent(
                _NUMBER, PhoneNumberAssignAgentRequest(agent_id="agt_1")
            )
        else:
            client = make_client([(204, None, None)], rec)
            client.phone_numbers.unassign_agent(_NUMBER)

        raw_path = rec.requests[0].url.raw_path.decode()
        assert _ENCODED in raw_path, (op, raw_path)
        # negative anchors: no literal +, no space, not the un-encoded path
        assert "+14155550132" not in raw_path, (op, raw_path)
        assert " " not in raw_path, (op, raw_path)
        assert raw_path.split("?")[0] != "/v1/phone-numbers/+14155550132", op


def test_opaque_ids_not_over_encoded(make_client):
    # session_id / subscription_id are opaque ids -> raw f-string, verbatim
    opaque = "abc-123_XYZ"

    rec_sess = Recorder()
    c1 = make_client([(200, _session_response_dict(opaque), None)], rec_sess)
    c1.sessions.get(opaque)
    assert rec_sess.requests[0].url.path == f"/v1/sessions/{opaque}"
    assert "%2D" not in rec_sess.requests[0].url.raw_path.decode()

    rec_hook = Recorder()
    c2 = make_client([(200, _webhook_subscription_dict(opaque), None)], rec_hook)
    c2.webhooks.get(opaque)
    assert rec_hook.requests[0].url.path == f"/v1/webhook-subscriptions/{opaque}"
    assert "%5F" not in rec_hook.requests[0].url.raw_path.decode()


# ===========================================================================
# idempotency selectivity (AC-8) — money safety
# ===========================================================================


def test_idempotency_key_present_on_three_phone_posts(make_client):
    # exactly purchase_available / purchase / import_ carry a v4 Idempotency-Key
    rec_avail = Recorder()
    c1 = make_client([(201, _phone_response_dict(), None)], rec_avail)
    c1.phone_numbers.purchase_available(_purchase_available_body())
    key = rec_avail.header(HEADER)
    assert key is not None and key != ""
    assert uuid.UUID(key).version == 4

    rec_purchase = Recorder()
    c2 = make_client([(201, None, None)], rec_purchase)
    c2.phone_numbers.purchase(PurchasePhoneNumberRequest(phone_number=_NUMBER))
    key2 = rec_purchase.header(HEADER)
    assert key2 is not None and key2 != ""
    assert uuid.UUID(key2).version == 4

    rec_import = Recorder()
    c3 = make_client([(201, None, None)], rec_import)
    c3.phone_numbers.import_(ImportPhoneNumberRequest(phone_number=_NUMBER))
    key3 = rec_import.header(HEADER)
    assert key3 is not None and key3 != ""
    assert uuid.UUID(key3).version == 4


def test_idempotency_key_absent_on_all_other_writes(make_client):
    # assign_agent (PUT, not guarded), unassign_agent, deregister, all webhook
    # writes, sessions.delete: NONE carry an Idempotency-Key.
    rec_assign = Recorder()
    make_client([(200, None, None)], rec_assign).phone_numbers.assign_agent(
        _NUMBER, PhoneNumberAssignAgentRequest(agent_id="agt_1")
    )
    assert rec_assign.header(HEADER) is None

    rec_unassign = Recorder()
    make_client([(204, None, None)], rec_unassign).phone_numbers.unassign_agent(_NUMBER)
    assert rec_unassign.header(HEADER) is None

    rec_dereg = Recorder()
    make_client([(204, None, None)], rec_dereg).phone_numbers.deregister(_NUMBER)
    assert rec_dereg.header(HEADER) is None

    rec_create = Recorder()
    make_client(
        [(201, _webhook_subscription_dict(), None)], rec_create
    ).webhooks.create(_create_webhook_body())
    assert rec_create.header(HEADER) is None

    rec_update = Recorder()
    make_client(
        [(200, _webhook_subscription_dict(), None)], rec_update
    ).webhooks.update("whs_1", UpdateWebhookSubscriptionRequest(enabled=False))
    assert rec_update.header(HEADER) is None

    rec_wdelete = Recorder()
    make_client([(204, None, None)], rec_wdelete).webhooks.delete("whs_1")
    assert rec_wdelete.header(HEADER) is None

    rec_sdelete = Recorder()
    make_client([(204, None, None)], rec_sdelete).sessions.delete("sess_1")
    assert rec_sdelete.header(HEADER) is None


# ===========================================================================
# empty-body ops return None without parse (AC-6)
# ===========================================================================


def test_empty_body_ops_return_none_without_parse(make_client):
    # Each empty-body op returns exactly None even when the transport hands back a
    # non-JSON / empty body. If any of these went through model_validate, a
    # non-JSON body would raise — so reaching the assertion proves no parse.
    # purchase / import_ / assign_agent send a typed body; the DELETEs do not.
    cases = [
        lambda c: c.phone_numbers.purchase(
            PurchasePhoneNumberRequest(phone_number=_NUMBER)
        ),
        lambda c: c.phone_numbers.import_(
            ImportPhoneNumberRequest(phone_number=_NUMBER)
        ),
        lambda c: c.phone_numbers.assign_agent(
            _NUMBER, PhoneNumberAssignAgentRequest(agent_id="agt_1")
        ),
        lambda c: c.phone_numbers.deregister(_NUMBER),
        lambda c: c.phone_numbers.unassign_agent(_NUMBER),
        lambda c: c.sessions.delete("sess_1"),
        lambda c: c.webhooks.delete("whs_1"),
    ]
    for call in cases:
        # non-JSON, non-empty body on a 200 — model_validate would raise on this
        rec = Recorder()
        client = make_client([(200, b"not json at all", None)], rec)
        assert call(client) is None


# ===========================================================================
# pager item types + cursor threading (AC-9)
# ===========================================================================


def test_pager_item_types_for_four_lists(make_client):
    specs = [
        (
            lambda c: c.phone_numbers.list(),
            {
                "phone_numbers": [_phone_response_dict("pn_1")],
                "response_metadata": {"next_cursor": ""},
            },
            PhoneNumberResponse,
        ),
        (
            lambda c: c.sessions.list(),
            {
                "sessions": [_session_list_item_dict("sess_1")],
                "response_metadata": {"next_cursor": ""},
            },
            SessionListItem,
        ),
        (
            lambda c: c.webhooks.list(),
            {
                "subscriptions": [_webhook_subscription_dict("whs_1")],
                "response_metadata": {"next_cursor": ""},
            },
            WebhookSubscriptionResponse,
        ),
        (
            lambda c: c.webhooks.list_deliveries("sess_1"),
            {
                "deliveries": [_webhook_delivery_dict("whd_1")],
                "response_metadata": {"next_cursor": ""},
            },
            WebhookDeliveryResponse,
        ),
    ]
    for make_pager, page, item_type in specs:
        rec = Recorder()
        client = make_client([page], rec)
        items = list(make_pager(client))
        assert len(items) == 1, item_type
        assert all(isinstance(i, item_type) for i in items), item_type


def test_two_page_cursor_threads_and_terminates(make_client, recorder: Recorder):
    client = make_client(
        [
            {
                "phone_numbers": [_phone_response_dict("pn_1")],
                "has_more": True,
                "response_metadata": {"next_cursor": "c1"},
            },
            {
                "phone_numbers": [_phone_response_dict("pn_2")],
                "has_more": False,
                "response_metadata": {"next_cursor": ""},
            },
        ],
        recorder,
    )

    items = list(client.phone_numbers.list(limit=1))

    assert [i.id for i in items] == ["pn_1", "pn_2"]
    assert recorder.count == 2
    # first page has no cursor; second carries the threaded cursor
    assert "cursor" not in dict(recorder.requests[0].url.params)
    assert dict(recorder.requests[1].url.params).get("cursor") == "c1"


# ===========================================================================
# list filters serialize (AC-10)
# ===========================================================================


def test_sessions_list_filters_serialize(make_client, recorder: Recorder):
    client = make_client(
        [{"sessions": [], "response_metadata": {"next_cursor": ""}}], recorder
    )

    list(client.sessions.list(status="completed", agent_id="ag_1", limit=5))

    q = dict(recorder.requests[0].url.params)
    assert q.get("status") == "completed"
    assert q.get("agent_id") == "ag_1"
    assert q.get("limit") == "5"

    # omitted filters absent
    rec2 = Recorder()
    c2 = make_client([{"sessions": [], "response_metadata": {"next_cursor": ""}}], rec2)
    list(c2.sessions.list(limit=5))
    q2 = dict(rec2.requests[0].url.params)
    assert "status" not in q2
    assert "agent_id" not in q2


def test_webhooks_list_include_disabled_filter(make_client, recorder: Recorder):
    client = make_client(
        [{"subscriptions": [], "response_metadata": {"next_cursor": ""}}], recorder
    )

    list(client.webhooks.list(include_disabled=True))

    q = dict(recorder.requests[0].url.params)
    # the param is present and renders to a truthy string form
    # (httpx lowercases a Python bool to "true"; a str-coerced impl gives "True").
    assert "include_disabled" in q
    assert q["include_disabled"].lower() in ("true", "1")

    # omitted -> absent
    rec2 = Recorder()
    c2 = make_client(
        [{"subscriptions": [], "response_metadata": {"next_cursor": ""}}], rec2
    )
    list(c2.webhooks.list())
    assert "include_disabled" not in dict(rec2.requests[0].url.params)


def test_limit_cursor_only_lists_reject_extra_kwargs(make_client):
    # phone_numbers.list and list_deliveries accept ONLY limit/cursor (keyword
    # only). Passing status/include_disabled is a TypeError.
    client = make_client(
        [{"phone_numbers": [], "response_metadata": {"next_cursor": ""}}], Recorder()
    )
    with pytest.raises(TypeError):
        client.phone_numbers.list(status="completed")
    with pytest.raises(TypeError):
        client.webhooks.list_deliveries("sess_1", include_disabled=True)

    # the accepted kwargs do work
    rec = Recorder()
    c2 = make_client(
        [{"phone_numbers": [], "response_metadata": {"next_cursor": ""}}], rec
    )
    list(c2.phone_numbers.list(limit=3, cursor="c0"))
    q = dict(rec.requests[0].url.params)
    assert q.get("limit") == "3"
    assert q.get("cursor") == "c0"


# ===========================================================================
# latest-deliveries takes no input beyond the path (AC-11)
# ===========================================================================


def test_list_latest_deliveries_sends_no_body_or_query(make_client, recorder: Recorder):
    client = make_client(
        [(200, {"deliveries": [], "response_metadata": {"next_cursor": ""}}, None)],
        recorder,
    )

    result = client.webhooks.list_latest_deliveries("sess_1")

    assert isinstance(result, WebhookDeliveryListResponse)
    req = recorder.requests[0]
    assert req.method == "GET"
    assert req.url.path == "/v1/sessions/sess_1/webhook-deliveries/latest"
    assert dict(req.url.params) == {}
    assert req.content in (b"", b"null"), req.content


# ===========================================================================
# webhooks.test() / test-fire absent (AC-12)
# ===========================================================================


def test_webhooks_test_fire_method_absent(make_client, make_async_client):
    sync = make_client([(200, {"ok": True}, None)], Recorder())
    asy = make_async_client([(200, {"ok": True}, None)], Recorder())
    for resource in (sync.webhooks, asy.webhooks):
        for attr in ("test", "test_fire", "fire"):
            assert not hasattr(resource, attr), attr
    # also on the classes directly
    for cls in (WebhooksResource, AsyncWebhooksResource):
        for attr in ("test", "test_fire", "fire"):
            assert not hasattr(cls, attr), (cls, attr)


# ===========================================================================
# eager accessors in both ctors (AC-15)
# ===========================================================================


def test_eager_accessors_in_both_ctors(make_client, make_async_client):
    sync = make_client([(200, {"ok": True}, None)], Recorder())
    assert isinstance(sync.phone_numbers, PhoneNumbersResource)
    assert isinstance(sync.sessions, SessionsResource)
    assert isinstance(sync.webhooks, WebhooksResource)
    # not the async variants
    assert not isinstance(sync.phone_numbers, AsyncPhoneNumbersResource)

    asy = make_async_client([(200, {"ok": True}, None)], Recorder())
    assert isinstance(asy.phone_numbers, AsyncPhoneNumbersResource)
    assert isinstance(asy.sessions, AsyncSessionsResource)
    assert isinstance(asy.webhooks, AsyncWebhooksResource)
    assert not isinstance(asy.phone_numbers, PhoneNumbersResource)


# ===========================================================================
# fail-at-base anchor (AC-16)
# ===========================================================================


def test_fail_at_base_namespaces_absent(make_client, make_async_client):
    # At base SHA the three namespaces do not exist; this asserts they DO at HEAD.
    # The module-level imports of PhoneNumbersResource etc. already fail to import
    # at base (ModuleNotFoundError), so the whole file errors there. This test
    # additionally pins the eager attribute presence on the client.
    sync = make_client([(200, {"ok": True}, None)], Recorder())
    assert hasattr(sync, "phone_numbers")
    assert hasattr(sync, "sessions")
    assert hasattr(sync, "webhooks")

    asy = make_async_client([(200, {"ok": True}, None)], Recorder())
    assert hasattr(asy, "phone_numbers")
    assert hasattr(asy, "sessions")
    assert hasattr(asy, "webhooks")


# ===========================================================================
# existing consumers unaffected
# ===========================================================================


def test_existing_client_consumers_unaffected(make_client, make_async_client):
    # the additive accessors must not disturb the PR13a surface
    from assemblyai_agents.resources.agents import (
        AgentsResource,
        AsyncAgentsResource,
    )
    from assemblyai_agents.resources.calls import (
        AsyncCallsResource,
        CallsResource,
    )
    from assemblyai_agents.resources.tokens import (
        AsyncTokensResource,
        TokensResource,
    )

    sync = make_client([(200, {"ok": True}, None)], Recorder())
    assert isinstance(sync.agents, AgentsResource)
    assert isinstance(sync.calls, CallsResource)
    assert isinstance(sync.tokens, TokensResource)
    # core methods still present and callable
    assert callable(sync.request)
    assert callable(sync.request_raw)
    assert callable(sync.paginate)

    asy = make_async_client([(200, {"ok": True}, None)], Recorder())
    assert isinstance(asy.agents, AsyncAgentsResource)
    assert isinstance(asy.calls, AsyncCallsResource)
    assert isinstance(asy.tokens, AsyncTokensResource)
