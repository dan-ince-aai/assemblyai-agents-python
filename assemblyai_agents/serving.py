"""Serve your own functions, so there is no backend to write.

Everything the platform reaches — a tool, a pre-connect lookup, the reply
endpoint — is a function on your declaration. `serve()` answers those requests
with those functions, on the standard library alone. `asgi()` hands the same
handlers to any ASGI host — Modal, uvicorn, a Lambda adapter — for platforms
that want an app rather than a process.

    agent.serve()                         # address → deploy → serve

    POST /tools/{name}            each hosted @tool on the declaration
    POST /v1/chat/completions     the agent's `reply`, streamed as the platform wants
    POST /pre-connect/{name}      each pre-connect handler
    POST /webhooks/voice-agents   verified against your secret
    GET  /healthz                 what is loaded

`routes()` hands back the same handlers as plain callables to mount anywhere.
Nothing here holds state between requests, so several copies are fine, and
nothing here knows what a tunnel is: the address comes from you.
"""

import asyncio
import json
import os
import re
import secrets as _secrets
import socket
import threading
import warnings
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Mapping, Optional

from . import webhooks as _webhooks
from ._exceptions import ConfigurationError
from .replies import Turn, json_body, stream

_JSON = "application/json"
ENV_PORT = "PORT"


def _print(message: str) -> None:
    """Print and flush, because a log you cannot see until the process exits is
    no use while a call is in progress."""
    print(message, flush=True)


class Refused(Exception):
    """A request that should not be answered, with the status to answer instead."""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail


class _Headers(Mapping[str, str]):
    """Case-insensitive view over request headers, so `Authorization` and
    `X-AAI-Signature` read the same from any server."""

    def __init__(self, items: Mapping[str, str]) -> None:
        self._items = {str(k).lower(): v for k, v in items.items()}

    def __getitem__(self, key: str) -> str:
        return self._items[key.lower()]

    def __iter__(self):
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)


def _coerce(arguments: dict, schema: dict) -> dict:
    """Restore types on a GET tool's query string, which is all strings."""
    properties = (schema or {}).get("properties", {})
    restored = {}
    for key, value in arguments.items():
        kind = properties.get(key, {}).get("type")
        try:
            if kind == "integer":
                value = int(value)
            elif kind == "number":
                value = float(value)
            elif kind == "boolean":
                value = str(value).lower() in ("1", "true", "yes")
        except (TypeError, ValueError):
            pass
        restored[key] = value
    return restored


def _secret_for(secret: Optional[str], legacy: Optional[str], name: str) -> Optional[str]:
    if legacy is not None:
        warnings.warn(
            f"serve({name}=...) is deprecated; pass secret=, which the platform presents "
            f"on every hosted route.",
            DeprecationWarning,
            stacklevel=4,
        )
        return legacy
    return secret


def routes(
    agent: Any,
    *,
    secret: Optional[str] = None,
    reply: Optional[Callable[[Turn], Any]] = None,
    pre_connect: Optional[Mapping[str, Callable]] = None,
    webhook_secret: Optional[str] = None,
    on_event: Optional[Callable[[dict], None]] = None,
    tool_secret: Optional[str] = None,
    llm_key: Optional[str] = None,
) -> dict:
    """The handlers, without a server, for mounting in your own framework.

    Each takes `(path, query, body_bytes, headers)` and returns
    `(status, payload)`, where a payload that is an iterator is streamed.

    Everything is read off the declaration: hosted tools, the `reply`
    function, pre-connect handlers. `reply=` and `pre_connect=` here override
    or add to what the declaration carries. `tool_secret=` / `llm_key=` are the
    deprecated spellings of `secret=`.
    """
    tool_secret = _secret_for(secret, tool_secret, "tool_secret")
    llm_key = _secret_for(secret, llm_key, "llm_key")
    reply = reply if reply is not None else getattr(agent, "reply", None)

    tools = {declared.name: declared for declared in agent.tools or [] if declared.hosted}

    handlers: dict = {}
    for entry in getattr(agent, "pre_connect", None) or []:
        if getattr(entry, "handler", None) is not None:
            handlers[entry.path] = entry.handler
    handlers.update(pre_connect or {})

    def check(headers: Mapping[str, str], expected: Optional[str], status: int = 401) -> None:
        if expected and headers.get("Authorization") != f"Bearer {expected}":
            raise Refused(status, "unauthorized")

    def run_tool(path: str, query: str, body: bytes, headers: Mapping[str, str]):
        check(headers, tool_secret)
        name = path.rsplit("/", 1)[-1]
        declared = tools.get(name)
        if declared is None:
            raise Refused(404, f"no tool named {name!r} is hosted here")
        if body:
            arguments = json.loads(body)
        else:
            # A GET or DELETE tool sends its arguments as a query string.
            from urllib.parse import parse_qsl

            arguments = _coerce(dict(parse_qsl(query)), declared.spec.parameters)
        if not isinstance(arguments, dict):
            raise Refused(422, "arguments were not a JSON object")
        try:
            return 200, asyncio.run(declared.invoke(**arguments))
        except TypeError as exc:
            raise Refused(422, str(exc)) from exc

    def replies(path: str, query: str, body: bytes, headers: Mapping[str, str]):
        if reply is None:
            raise Refused(404, "this agent has no reply function; the platform's model talks")
        check(headers, llm_key)
        request = json.loads(body or b"{}")
        turn = Turn.from_request(request)
        answer = reply(turn)
        if request.get("stream"):
            return 200, stream(turn, answer)
        return 200, json_body(turn, answer)

    def webhook(path: str, query: str, body: bytes, headers: Mapping[str, str]):
        if not webhook_secret:
            raise Refused(503, "no webhook secret configured")
        try:
            event = _webhooks.verify(body, headers.get("X-AAI-Signature", ""), webhook_secret)
        except _webhooks.WebhookVerificationError as exc:
            raise Refused(400, str(exc)) from exc
        if on_event is not None:
            on_event(event)
        return 204, None

    def health(path: str, query: str, body: bytes, headers: Mapping[str, str]):
        return 200, {
            "ok": True,
            "agent": agent.name,
            "tools": sorted(tools),
            "reply": reply is not None,
            "pre_connect": sorted(handlers),
        }

    table: dict = {
        ("POST", re.compile(r"^/tools/[^/]+$")): run_tool,
        ("POST", re.compile(r"^/v1/chat/completions$")): replies,
        ("POST", re.compile(r"^/webhooks/voice-agents$")): webhook,
        ("GET", re.compile(r"^/healthz$")): health,
    }

    for hook_path, handler in handlers.items():
        def wrap(handler=handler):
            def run(path: str, query: str, body: bytes, headers: Mapping[str, str]):
                check(headers, tool_secret)
                payload = json.loads(body) if body else {}
                result = handler(payload)
                if asyncio.iscoroutine(result):
                    result = asyncio.run(result)
                return 200, result

            return run

        table[("POST", re.compile(rf"^{re.escape(hook_path)}$"))] = wrap()

    return table


def _dispatch(table: dict, method: str, path: str, query: str, body: bytes,
              headers: Mapping[str, str], note: Callable[[str], None]):
    """Run the matching handler; returns (status, payload) or (status, json dict)."""
    for (route_method, pattern), handler in table.items():
        if route_method != method or not pattern.match(path):
            continue
        try:
            return handler(path, query, body, headers)
        except Refused as refused:
            note(f"{method} {path} -> {refused.status} {refused.detail}")
            return refused.status, {"detail": refused.detail}
        except Exception as exc:  # a tool raised: say so, keep serving
            note(f"{method} {path} -> 500 {type(exc).__name__}: {exc}")
            return 500, {"detail": f"{type(exc).__name__}: {exc}"}
    return 404, {"detail": f"no route for {method} {path}"}


def _is_stream(payload: Any) -> bool:
    return hasattr(payload, "__iter__") and not isinstance(payload, (dict, list, str, bytes))


def claim_port(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Fail now if the port is taken, before anything irreversible happens.

    Deploy is the step that repoints a stored agent, and it runs before
    `serve()` binds — so without this, a second run of a script whose first run
    is still holding the port updates the live agent and only then dies, leaving
    a reachable agent whose tool URLs answer to nothing. Call this first.
    """
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind((host, port))
    except OSError as exc:
        raise OSError(
            f"cannot serve on {host}:{port}: {exc}. Something else is holding it — "
            f"usually an earlier run of this script, which on macOS is a process "
            f"named `Python`, not `python`. Stop it (`lsof -ti :{port} | xargs kill`), "
            f"or set PORT to a free one."
        ) from exc
    finally:
        probe.close()


def serve(
    agent: Any,
    *,
    secret: Optional[str] = None,
    host: str = "0.0.0.0",
    port: Optional[int] = None,
    reply: Optional[Callable[[Turn], Any]] = None,
    pre_connect: Optional[Mapping[str, Callable]] = None,
    webhook_secret: Optional[str] = None,
    on_event: Optional[Callable[[dict], None]] = None,
    log: Optional[Callable[[str], None]] = _print,
    background: bool = False,
    tool_secret: Optional[str] = None,
    llm_key: Optional[str] = None,
) -> Any:
    """Answer the platform's HTTPS requests with your own functions.

    Blocks until interrupted. With `background=True` it returns the server, for
    a test or a script that has other work to do; call `.shutdown()` when done.
    `port` defaults to the PORT environment variable, then 8000.

    This only serves. `agent.serve()` is the whole flow — it resolves the
    address, deploys, and then calls this.
    """
    port = port if port is not None else int(os.environ.get(ENV_PORT, "8000"))
    table = routes(
        agent,
        secret=secret,
        reply=reply,
        pre_connect=pre_connect,
        webhook_secret=webhook_secret,
        on_event=on_event,
        tool_secret=tool_secret,
        llm_key=llm_key,
    )

    def note(message: str) -> None:
        if log is not None:
            log(message)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "assemblyai-agents"

        def log_message(self, *args) -> None:
            pass  # the interesting lines are logged below, in our own words

        def _handle(self) -> None:
            from urllib.parse import urlsplit

            parts = urlsplit(self.path)
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else b""
            status, payload = _dispatch(
                table, self.command, parts.path, parts.query, body, _Headers(self.headers), note
            )
            if _is_stream(payload):
                note(f"{self.command} {parts.path} -> streaming")
                self._send_stream(payload)
                return
            if status < 400:
                note(f"{self.command} {parts.path} -> {status}")
            self._send_json(status, payload)

        do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = _handle

        def _send_json(self, status: int, payload: Any) -> None:
            encoded = b"" if payload is None else json.dumps(payload, default=str).encode()
            self.send_response(status)
            if encoded:
                self.send_header("Content-Type", _JSON)
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            if encoded:
                self.wfile.write(encoded)

        def _send_stream(self, chunks) -> None:
            # Server-Sent Events, chunked, so the platform can start speaking
            # before the whole answer exists.
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            try:
                for chunk in chunks:
                    encoded = chunk.encode()
                    self.wfile.write(f"{len(encoded):X}\r\n".encode() + encoded + b"\r\n")
                    self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass  # the platform hung up mid-answer, which is its business

    try:
        server = ThreadingHTTPServer((host, port), Handler)
    except OSError as exc:
        raise OSError(
            f"cannot serve on {host}:{port}: {exc}. Something else is holding it — "
            f"usually an earlier run of this script. Stop it (`lsof -ti :{port} | "
            f"xargs kill`), or set PORT to a free one."
        ) from exc
    note(f"serving {agent.name!r} on http://{host}:{port} — {', '.join(sorted(r[1].pattern for r in table))}")
    if background:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return server


def asgi(
    agent: Any,
    *,
    secret: Optional[str] = None,
    reply: Optional[Callable[[Turn], Any]] = None,
    pre_connect: Optional[Mapping[str, Callable]] = None,
    webhook_secret: Optional[str] = None,
    on_event: Optional[Callable[[dict], None]] = None,
    log: Optional[Callable[[str], None]] = _print,
) -> Callable:
    """The same routes as an ASGI application, for hosts that want an app.

        # Modal
        @app.function(image=image, secrets=[modal.Secret.from_name("assemblyai")])
        @modal.asgi_app()
        def web():
            return asgi(agent, secret=os.environ["AGENT_SECRET"])

        # uvicorn
        app = asgi(agent, secret=SECRET)        # uvicorn module:app --port $PORT

        # AWS Lambda, via Mangum
        handler = Mangum(asgi(agent, secret=SECRET))

    Handlers run in a worker thread, so a tool may be sync or async and the
    event loop the host owns is never blocked. Streaming replies are sent as
    they are produced.
    """
    table = routes(
        agent,
        secret=secret,
        reply=reply,
        pre_connect=pre_connect,
        webhook_secret=webhook_secret,
        on_event=on_event,
    )

    def note(message: str) -> None:
        if log is not None:
            log(message)

    async def app(scope: dict, receive: Callable, send: Callable) -> None:
        if scope["type"] == "lifespan":
            while True:
                message = await receive()
                if message["type"] == "lifespan.startup":
                    await send({"type": "lifespan.startup.complete"})
                elif message["type"] == "lifespan.shutdown":
                    await send({"type": "lifespan.shutdown.complete"})
                    return
        if scope["type"] != "http":
            return

        body = b""
        while True:
            message = await receive()
            body += message.get("body", b"")
            if not message.get("more_body"):
                break

        headers = _Headers({k.decode("latin-1"): v.decode("latin-1") for k, v in scope.get("headers", [])})
        path = scope["path"]
        query = scope.get("query_string", b"").decode()
        # Off the loop: a handler may call asyncio.run() for an async tool, and a
        # sync tool may take its whole timeout.
        status, payload = await asyncio.to_thread(
            _dispatch, table, scope["method"], path, query, body, headers, note
        )

        if _is_stream(payload):
            note(f"{scope['method']} {path} -> streaming")
            await send({"type": "http.response.start", "status": 200, "headers": [
                (b"content-type", b"text/event-stream"), (b"cache-control", b"no-cache"),
            ]})
            for chunk in payload:
                await send({"type": "http.response.body", "body": chunk.encode(), "more_body": True})
            await send({"type": "http.response.body", "body": b"", "more_body": False})
            return

        if status < 400:
            note(f"{scope['method']} {path} -> {status}")
        encoded = b"" if payload is None else json.dumps(payload, default=str).encode()
        response_headers = [(b"content-length", str(len(encoded)).encode())]
        if encoded:
            response_headers.insert(0, (b"content-type", _JSON.encode()))
        await send({"type": "http.response.start", "status": status, "headers": response_headers})
        await send({"type": "http.response.body", "body": encoded})

    return app


def serve_agent(
    agent: Any,
    *,
    public_url: Optional[str] = None,
    secret: Optional[str] = None,
    host: str = "0.0.0.0",
    port: Optional[int] = None,
    deploy: bool = True,
    client: Any = None,
    agent_id: Optional[str] = None,
    id_file: Optional[str] = ".agent_id",
    webhook_secret: Optional[str] = None,
    on_event: Optional[Callable[[dict], None]] = None,
    log: Optional[Callable[[str], None]] = _print,
    background: bool = False,
) -> Any:
    """Address → deploy → serve. What `agent.serve()` runs.

    1. Resolve the address (`public_url` or PUBLIC_BASE_URL) and the secret
       (`secret` or AGENT_SECRET, else one is minted for this run).
    2. Claim the port, so a second run cannot repoint a live agent and then die.
    3. Deploy the declaration bound to that address (create, or update the
       stored id), unless `deploy=False`.
    4. Serve.
    """
    from .deploy import deploy as _deploy, resolve_public_url, resolve_secret

    port = port if port is not None else int(os.environ.get(ENV_PORT, "8000"))
    secret = resolve_secret(secret) or _secrets.token_urlsafe(32)
    bound = agent
    if agent.needs_address:
        bound = agent.hosted_at(agent.public_url or resolve_public_url(public_url, verb="serve"), secret=secret)

    claim_port(host, port)
    if deploy:
        _deploy(bound, client=client, agent_id=agent_id, id_file=id_file, log=log)
    return serve(
        bound,
        secret=secret,
        host=host,
        port=port,
        webhook_secret=webhook_secret,
        on_event=on_event,
        log=log,
        background=background,
    )
