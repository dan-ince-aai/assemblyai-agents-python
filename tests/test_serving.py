"""`serve()` against a real socket, in the shapes the platform sends.

Each case is a request the platform can make, or an answer it reads in a way
that is easy to get wrong. The comments say which.
"""

import hashlib
import hmac
import json
import time
import urllib.error
import urllib.request
from types import SimpleNamespace

import pytest
from assemblyai_agents import ToolContext, VoiceAgent, tool
from assemblyai_agents.byo import say
from assemblyai_agents.serving import PRE_CONNECT_RESPONSE_LIMIT_BYTES, serve
from pydantic import BaseModel

SECRET = "s3cret"
WEBHOOK_SECRET = "whsec_" + "a" * 40


class Order(BaseModel):
    order_id: str
    total: int


@tool
def lookup_order(order_id: str, count: int = 1, gift: bool = False) -> dict:
    """Look up one of the caller's orders."""
    return {"order_id": order_id, "count": count, "gift": gift}


@tool
def greet(name: str) -> str:
    """Say hello."""
    return f"hello {name}"


@tool
def fetch_order(order_id: str) -> Order:
    """Fetch an order as a model."""
    return Order(order_id=order_id, total=12)


@tool
def explode(order_id: str) -> dict:
    """Always fail."""
    raise RuntimeError("token=sk-live-123 leaked in a message")


@tool
def whoami(context: ToolContext) -> dict:
    """Say which session this is."""
    return {"session_id": context.session_id}


AGENT = VoiceAgent(
    name="Test Line",
    voice="ivy",
    system_prompt="prompt",
    tools=[lookup_order, greet, fetch_order, explode, whoami],
)


@pytest.fixture
def served():
    events = []
    lines = []
    server = serve(
        AGENT,
        reply=lambda turn: say(f"you said {turn.caller_said}"),
        host="127.0.0.1",
        port=0,
        tool_secret=SECRET,
        llm_key="llm-key",
        pre_connect={
            "/pre-connect": lambda payload: {
                "greeting": f"Hi {payload.get('caller_number')}"
            },
            "/nothing": lambda payload: None,
            "/huge": lambda payload: {"blob": "x" * PRE_CONNECT_RESPONSE_LIMIT_BYTES},
        },
        webhook_secret=WEBHOOK_SECRET,
        on_event=events.append,
        context=SimpleNamespace(session_id="sess_1"),
        log=lines.append,
        background=True,
    )
    base = f"http://127.0.0.1:{server.server_address[1]}"
    yield base, events, lines
    server.shutdown()
    server.server_close()


def call(base, path, *, method="POST", body=None, token=SECRET, headers=None):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(base + path, data=data, method=method)
    if token is not None:
        request.add_header("Authorization", f"Bearer {token}")
    if data is not None:
        request.add_header("Content-Type", "application/json")
    for name, value in (headers or {}).items():
        request.add_header(name, value)
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return (
                response.status,
                response.headers.get("Content-Type"),
                response.read(),
            )
    except urllib.error.HTTPError as error:
        return error.code, error.headers.get("Content-Type"), error.read()


def test_a_tool_answers_its_json_body(served):
    base, _, _ = served
    status, _, body = call(
        base, "/tools/lookup_order", body={"order_id": "W1", "count": 2}
    )
    assert status == 200
    assert json.loads(body) == {"order_id": "W1", "count": 2, "gift": False}


def test_a_get_tool_reads_and_types_its_query_string(served):
    # The platform sends a GET tool's arguments as str(value), so True arrives
    # as "True".
    base, _, _ = served
    status, _, body = call(
        base, "/tools/lookup_order?order_id=W1&count=3&gift=True", method="GET"
    )
    assert status == 200
    assert json.loads(body) == {"order_id": "W1", "count": 3, "gift": True}


def test_a_string_result_is_sent_unquoted(served):
    base, _, _ = served
    status, content_type, body = call(base, "/tools/greet", body={"name": "Maria"})
    assert status == 200
    assert content_type.startswith("text/plain")
    assert body == b"hello Maria"


def test_a_model_result_is_sent_as_its_json(served):
    base, _, _ = served
    _, _, body = call(base, "/tools/fetch_order", body={"order_id": "W1"})
    assert json.loads(body) == {"order_id": "W1", "total": 12}


def test_a_raising_tool_never_sends_its_message(served):
    # The platform hands a tool's error body to the model, which speaks it.
    base, _, lines = served
    status, _, body = call(base, "/tools/explode", body={"order_id": "W1"})
    assert status == 500
    assert json.loads(body) == {"error": "tool_raised", "type": "RuntimeError"}
    assert any("sk-live-123" in line for line in lines)


def test_a_tool_that_takes_a_context_is_handed_one(served):
    base, _, _ = served
    status, _, body = call(base, "/tools/whoami", body={})
    assert status == 200
    assert json.loads(body) == {"session_id": "sess_1"}


def test_a_bad_argument_is_refused_before_the_tool_runs(served):
    base, _, _ = served
    status, _, _ = call(base, "/tools/lookup_order", body={"order_id": "W1", "count": "many"})
    assert status == 422


def test_a_tool_request_without_the_secret_is_refused(served):
    base, _, _ = served
    status, _, _ = call(
        base, "/tools/lookup_order", body={"order_id": "W1"}, token="wrong"
    )
    assert status == 401


def test_an_unknown_tool_is_not_found(served):
    base, _, _ = served
    status, _, _ = call(base, "/tools/missing", body={})
    assert status == 404


@pytest.mark.parametrize(
    "path", ["/v1/chat/completions", "/chat/completions", "/llm/v2/chat/completions"]
)
def test_the_reply_endpoint_streams_under_any_base_url(served, path):
    # The OpenAI client posts to `<base_url>/chat/completions`.
    base, _, _ = served
    request = {
        "model": "engine",
        "stream": True,
        "stream_options": {"include_usage": True},
        "messages": [
            {"role": "system", "content": "prompt"},
            {"role": "user", "content": "hello"},
        ],
    }
    status, content_type, body = call(base, path, body=request, token="llm-key")
    assert status == 200
    assert content_type == "text/event-stream"
    frames = [line[len("data: ") :] for line in body.decode().split("\n\n") if line]
    assert frames[-1] == "[DONE]"
    words = "".join(
        chunk["choices"][0]["delta"].get("content") or ""
        for chunk in map(json.loads, frames[:-1])
        if chunk["choices"]
    )
    assert words.strip() == "you said hello"


def test_the_reply_endpoint_checks_the_llm_key(served):
    base, _, _ = served
    status, _, _ = call(
        base, "/v1/chat/completions", body={"messages": []}, token="wrong"
    )
    assert status == 401


@pytest.mark.parametrize("method", ["POST", "GET"])
def test_pre_connect_answers_with_the_handlers_body(served, method):
    # The platform sends a GET entry's values as a query string.
    base, _, _ = served
    if method == "GET":
        status, _, body = call(
            base, "/pre-connect?caller_number=%2B15550100", method="GET"
        )
    else:
        status, _, body = call(
            base, "/pre-connect", body={"caller_number": "+15550100"}
        )
    assert status == 200
    assert json.loads(body) == {"greeting": "Hi +15550100"}


def test_a_pre_connect_handler_returning_nothing_still_sends_json(served):
    # An empty body fails the entry, and an entry set to reject refuses the call.
    base, _, _ = served
    status, _, body = call(base, "/nothing", body={})
    assert status == 200
    assert json.loads(body) == {}


def test_an_oversized_pre_connect_answer_is_logged(served):
    base, _, lines = served
    call(base, "/huge", body={})
    assert any(
        "/huge" in line and str(PRE_CONNECT_RESPONSE_LIMIT_BYTES) in line
        for line in lines
    )


def _signed(body: bytes) -> str:
    timestamp = int(time.time())
    mac = hmac.new(
        WEBHOOK_SECRET.encode(), f"{timestamp}.".encode() + body, hashlib.sha256
    ).hexdigest()
    return f"t={timestamp},v1={mac}"


def test_a_signed_webhook_is_delivered(served):
    base, events, _ = served
    raw = json.dumps({"event": "session.completed"}).encode()
    request = urllib.request.Request(
        base + "/webhooks/voice-agents",
        data=raw,
        method="POST",
        headers={"X-AAI-Signature": _signed(raw)},
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        assert response.status == 204
    assert events == [{"event": "session.completed"}]


def test_a_badly_signed_webhook_is_refused(served):
    base, events, _ = served
    status, _, _ = call(
        base,
        "/webhooks/voice-agents",
        body={"event": "session.completed"},
        token=None,
        headers={"X-AAI-Signature": "t=1,v1=00"},
    )
    assert status == 400
    assert events == []


def test_health_says_what_is_loaded(served):
    base, _, _ = served
    status, _, body = call(base, "/healthz", method="GET", token=None)
    assert status == 200
    assert json.loads(body) == {
        "ok": True,
        "agent": "Test Line",
        "tools": ["explode", "fetch_order", "greet", "lookup_order", "whoami"],
        "reply_endpoint": True,
        "pre_connect": ["/huge", "/nothing", "/pre-connect"],
    }


def test_startup_logs_each_route_once(served):
    _, _, lines = served
    startup = next(line for line in lines if line.startswith("serving 'Test Line'"))
    # Tools and pre-connect paths answer on every method; listed once each.
    assert startup.count("/tools/{name}") == 1
    assert startup.count("/pre-connect") == 1
    assert "/chat/completions" in startup
