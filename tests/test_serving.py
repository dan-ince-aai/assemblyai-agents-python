"""serving: the routes, the stdlib server, and the ASGI app answer from the declaration."""

import asyncio
import json
import urllib.error
import urllib.request

import pytest

from assemblyai_agents import Captured, PreConnectRequest, VoiceAgent, tool
from assemblyai_agents.replies import Turn, call_tool, say
from assemblyai_agents.serving import Refused, asgi, routes, serve

SECRET = "s3cret"
AUTH = {"Authorization": f"Bearer {SECRET}", "Content-Type": "application/json"}


@tool(timeout_seconds=8)
async def check_stock(item_said: str) -> dict:
    """Check whether the shop has something.

    Args:
        item_said: What the caller asked for.
    """
    return {"found": True, "item": item_said}


@tool(url="https://api.example.com/weather")
def weather(city: str) -> dict:
    """Weather for a city, served elsewhere."""
    return {"city": city}


@tool(timeout_seconds=8)
def refuses(order_id: str) -> dict:
    """A tool that refuses bad input."""
    raise Refused(422, "order numbers start with W")


def lookup(payload: dict) -> dict:
    return {"matched": True, "greeting": "Hi again."}


def decide(turn: Turn):
    if turn.pending:
        return say(f"It is {turn.pending.get('item')}.")
    return call_tool("check_stock", item_said=turn.caller_said or "twine")


def make_agent(**overrides) -> VoiceAgent:
    fields = dict(
        name="Ridgeway Hardware",
        voice="alba",
        system_prompt="You answer the phone.",
        tools=[check_stock, weather, refuses],
        reply=decide,
        pre_connect=[PreConnectRequest(handler=lookup, returns=[Captured(name="ref", path="ref")], allow_overrides=True)],
    )
    fields.update(overrides)
    return VoiceAgent(**fields)


# --------------------------------------------------------------------------- routes


def test_routes_are_read_off_the_declaration():
    import re

    table = routes(make_agent(), secret=SECRET)
    patterns = sorted(pattern.pattern for _, pattern in table)
    assert patterns == [
        "^/healthz$",
        f"^{re.escape('/pre-connect/lookup')}$",      # a handler path is escaped verbatim
        "^/tools/[^/]+$",
        "^/v1/chat/completions$",
        "^/webhooks/voice-agents$",
    ]


def test_health_reports_only_what_this_process_hosts():
    table = routes(make_agent(), secret=SECRET)
    handler = next(h for (m, p), h in table.items() if p.pattern == "^/healthz$")
    status, payload = handler("/healthz", "", b"", {})
    assert status == 200
    # `weather` has its own URL, so it is not served here and not listed here.
    assert payload["tools"] == ["check_stock", "refuses"]
    assert payload["reply"] is True
    assert payload["pre_connect"] == ["/pre-connect/lookup"]


def test_an_agent_with_no_reply_has_no_reply_route_answer():
    table = routes(make_agent(reply=None), secret=SECRET)
    handler = next(h for (m, p), h in table.items() if p.pattern == "^/v1/chat/completions$")
    with pytest.raises(Refused) as exc:
        handler("/v1/chat/completions", "", b"{}", {"Authorization": f"Bearer {SECRET}"})
    assert exc.value.status == 404


def test_the_old_secret_spellings_still_work_and_warn():
    with pytest.warns(DeprecationWarning, match="tool_secret"):
        table = routes(make_agent(), tool_secret=SECRET, llm_key=SECRET)
    handler = next(h for (m, p), h in table.items() if p.pattern == "^/tools/[^/]+$")
    with pytest.raises(Refused) as exc:
        handler("/tools/check_stock", "", b"{}", {})
    assert exc.value.status == 401


# --------------------------------------------------------------------------- the stdlib server


@pytest.fixture
def server():
    srv = serve(make_agent(), secret=SECRET, port=8199, background=True, log=None)
    yield "http://127.0.0.1:8199"
    srv.shutdown()
    srv.server_close()


def _post(base, path, body: dict | None = None, headers=None):
    data = json.dumps(body).encode() if body is not None else b""
    request = urllib.request.Request(base + path, data=data, headers=headers or {}, method="POST")
    try:
        with urllib.request.urlopen(request) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def test_serve_runs_a_hosted_tool(server):
    status, body = _post(server, "/tools/check_stock", {"item_said": "glue"}, AUTH)
    assert (status, json.loads(body)) == (200, {"found": True, "item": "glue"})


def test_serve_checks_the_secret(server):
    assert _post(server, "/tools/check_stock", {"item_said": "glue"})[0] == 401
    assert _post(server, "/tools/check_stock", {"item_said": "glue"},
                 {**AUTH, "Authorization": "Bearer wrong"})[0] == 401


def test_serve_does_not_host_an_external_tool(server):
    status, body = _post(server, "/tools/weather", {"city": "Leeds"}, AUTH)
    assert status == 404
    assert json.loads(body)["detail"] == "no tool named 'weather' is hosted here"


def test_a_refused_call_answers_with_its_status(server):
    status, body = _post(server, "/tools/refuses", {"order_id": "X1"}, AUTH)
    assert (status, json.loads(body)["detail"]) == (422, "order numbers start with W")


def test_wrong_arguments_are_a_422_not_a_500(server):
    status, _ = _post(server, "/tools/check_stock", {"nope": 1}, AUTH)
    assert status == 422


def test_serve_answers_the_pre_connect_handler(server):
    status, body = _post(server, "/pre-connect/lookup", {}, AUTH)
    assert (status, json.loads(body)) == (200, {"matched": True, "greeting": "Hi again."})


def test_serve_streams_the_reply(server):
    request = urllib.request.Request(
        server + "/v1/chat/completions",
        data=json.dumps({"stream": True, "messages": [{"role": "system", "content": "x"},
                                                       {"role": "user", "content": "twine"}]}).encode(),
        headers=AUTH, method="POST",
    )
    with urllib.request.urlopen(request) as response:
        assert response.headers["Content-Type"] == "text/event-stream"
        body = response.read().decode()
    frames = [json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: ") and line != "data: [DONE]"]
    assert frames[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"] == "check_stock"
    assert body.rstrip().endswith("data: [DONE]")


def test_serve_health(server):
    with urllib.request.urlopen(server + "/healthz") as response:
        payload = json.loads(response.read())
    assert payload["ok"] is True and payload["agent"] == "Ridgeway Hardware"


# --------------------------------------------------------------------------- the ASGI app


async def _asgi_call(app, method, path, body=b"", headers=None):
    scope = {
        "type": "http", "method": method, "path": path, "query_string": b"",
        "headers": [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()],
    }
    sent = []
    messages = iter([{"type": "http.request", "body": body, "more_body": False}])

    async def receive():
        return next(messages)

    async def send(message):
        sent.append(message)

    await app(scope, receive, send)
    return sent[0]["status"], b"".join(m.get("body", b"") for m in sent[1:]), sent


def test_asgi_serves_the_same_routes():
    app = asgi(make_agent(), secret=SECRET, log=None)
    status, body, _ = asyncio.run(_asgi_call(app, "POST", "/tools/check_stock", json.dumps({"item_said": "twine"}).encode(), AUTH))
    assert (status, json.loads(body)) == (200, {"found": True, "item": "twine"})
    assert asyncio.run(_asgi_call(app, "POST", "/tools/check_stock", b"{}"))[0] == 401
    assert asyncio.run(_asgi_call(app, "POST", "/tools/weather", b"{}", AUTH))[0] == 404
    status, body, _ = asyncio.run(_asgi_call(app, "POST", "/pre-connect/lookup", b"{}", AUTH))
    assert (status, json.loads(body)) == (200, {"matched": True, "greeting": "Hi again."})


def test_asgi_streams_the_reply_as_server_sent_events():
    app = asgi(make_agent(), secret=SECRET, log=None)
    body = json.dumps({"stream": True, "messages": [{"role": "system", "content": "x"}, {"role": "user", "content": "twine"}]}).encode()
    status, data, sent = asyncio.run(_asgi_call(app, "POST", "/v1/chat/completions", body, AUTH))
    assert status == 200
    assert (b"content-type", b"text/event-stream") in sent[0]["headers"]
    # One frame per SSE chunk, then a terminating empty body.
    assert sum(1 for m in sent[1:] if m.get("more_body")) >= 2
    assert sent[-1] == {"type": "http.response.body", "body": b"", "more_body": False}
    assert data.decode().rstrip().endswith("data: [DONE]")


def test_asgi_headers_are_read_case_insensitively():
    app = asgi(make_agent(), secret=SECRET, log=None)
    status, _, _ = asyncio.run(_asgi_call(app, "POST", "/tools/check_stock", json.dumps({"item_said": "x"}).encode(),
                                          {"authorization": f"Bearer {SECRET}"}))
    assert status == 200


def test_asgi_answers_lifespan():
    app = asgi(make_agent(), secret=SECRET, log=None)
    messages = iter([{"type": "lifespan.startup"}, {"type": "lifespan.shutdown"}])
    out = []

    async def receive():
        return next(messages)

    async def send(message):
        out.append(message["type"])

    asyncio.run(app({"type": "lifespan"}, receive, send))
    assert out == ["lifespan.startup.complete", "lifespan.shutdown.complete"]
