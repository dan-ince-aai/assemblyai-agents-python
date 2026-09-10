"""Serve your own functions, so there is no backend to write.

Some things the platform can only reach over HTTPS: a tool that has to answer a
phone call, a pre-connect lookup, and your own reply endpoint. That does not
mean you should be writing a web service. `serve()` takes the agent you already
declared and answers those requests with the functions you already wrote.

    from assemblyai_agents.serving import serve

    serve(agent, reply=decide, tool_secret=SECRET, llm_key=KEY)

It uses the standard library, so there is nothing new to install, and it holds
no state, so it is safe to run several copies. Where it runs is not its
business: a laptop with a tunnel in front of it today, a container somewhere
tomorrow, without the code above changing.

    POST /tools/{name}            each @tool on the declaration
    POST /v1/chat/completions     `reply(turn)`, streamed as the platform wants
    POST <your pre-connect path>  a handler you pass in
    POST /webhooks/voice-agents   verified against your secret
    GET  /healthz                 what is loaded

For anything you would rather host yourself, `routes()` hands back the same
handlers as plain callables to mount wherever you like.
"""

import asyncio
import json
import re
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Mapping, Optional
from urllib.parse import parse_qsl, urlsplit

from . import webhooks as _webhooks
from .byo import Turn, json_body, stream

_JSON = "application/json"


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


def routes(
    agent: Any,
    *,
    reply: Optional[Callable[[Turn], Any]] = None,
    tool_secret: Optional[str] = None,
    llm_key: Optional[str] = None,
    pre_connect: Optional[Mapping[str, Callable]] = None,
    webhook_secret: Optional[str] = None,
    on_event: Optional[Callable[[dict], None]] = None,
) -> dict:
    """The handlers, without a server, for mounting in your own framework.

    Each takes `(path, query, body_bytes, headers)` and returns
    `(status, payload)`, where a payload that is an iterator is streamed.
    """
    tools = {declared.name: declared for declared in agent.tools or []}

    def check(headers: Mapping[str, str], secret: Optional[str], status: int = 401) -> None:
        if secret and headers.get("Authorization") != f"Bearer {secret}":
            raise Refused(status, "unauthorized")

    def run_tool(path: str, query: str, body: bytes, headers: Mapping[str, str]):
        check(headers, tool_secret)
        name = path.rsplit("/", 1)[-1]
        declared = tools.get(name)
        if declared is None:
            raise Refused(404, f"no tool named {name!r}")
        if body:
            arguments = json.loads(body)
        else:
            # A GET or DELETE tool sends its arguments as a query string.
            arguments = _coerce(dict(parse_qsl(query)), declared.spec.parameters)
        if not isinstance(arguments, dict):
            raise Refused(422, "arguments were not a JSON object")
        try:
            return 200, asyncio.run(declared.invoke(**arguments))
        except TypeError as exc:
            raise Refused(422, str(exc)) from exc

    def replies(path: str, query: str, body: bytes, headers: Mapping[str, str]):
        if reply is None:
            raise Refused(404, "this agent has no reply endpoint")
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
            "reply_endpoint": reply is not None,
            "pre_connect": sorted(pre_connect or {}),
        }

    table: dict = {
        ("POST", re.compile(r"^/tools/[^/]+$")): run_tool,
        ("POST", re.compile(r"^/v1/chat/completions$")): replies,
        ("POST", re.compile(r"^/webhooks/voice-agents$")): webhook,
        ("GET", re.compile(r"^/healthz$")): health,
    }

    for path, handler in (pre_connect or {}).items():
        def wrap(handler=handler):
            def run(path: str, query: str, body: bytes, headers: Mapping[str, str]):
                check(headers, tool_secret)
                payload = json.loads(body) if body else {}
                result = handler(payload)
                if asyncio.iscoroutine(result):
                    result = asyncio.run(result)
                return 200, result

            return run

        table[("POST", re.compile(rf"^{re.escape(path)}$"))] = wrap()

    return table


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
    reply: Optional[Callable[[Turn], Any]] = None,
    host: str = "0.0.0.0",
    port: int = 8000,
    tool_secret: Optional[str] = None,
    llm_key: Optional[str] = None,
    pre_connect: Optional[Mapping[str, Callable]] = None,
    webhook_secret: Optional[str] = None,
    on_event: Optional[Callable[[dict], None]] = None,
    log: Optional[Callable[[str], None]] = _print,
    background: bool = False,
) -> Any:
    """Answer the platform's HTTPS requests with your own functions.

    Blocks until interrupted. With `background=True` it returns the server, for
    a test or a script that has other work to do; call `.shutdown()` when done.
    """
    table = routes(
        agent,
        reply=reply,
        tool_secret=tool_secret,
        llm_key=llm_key,
        pre_connect=pre_connect,
        webhook_secret=webhook_secret,
        on_event=on_event,
    )

    def note(message: str) -> None:
        if log is not None:
            log(message)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "assemblyai-agents"

        def log_message(self, *args) -> None:
            pass  # the interesting lines are logged below, in our own words

        def _dispatch(self) -> None:
            parts = urlsplit(self.path)
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else b""
            for (method, pattern), handler in table.items():
                if method != self.command or not pattern.match(parts.path):
                    continue
                try:
                    status, payload = handler(parts.path, parts.query, body, self.headers)
                except Refused as refused:
                    note(f"{self.command} {parts.path} -> {refused.status} {refused.detail}")
                    self._send_json(refused.status, {"detail": refused.detail})
                    return
                except Exception as exc:  # a tool raised: say so, keep serving
                    note(f"{self.command} {parts.path} -> 500 {type(exc).__name__}: {exc}")
                    self._send_json(500, {"detail": f"{type(exc).__name__}: {exc}"})
                    return
                if hasattr(payload, "__iter__") and not isinstance(payload, (dict, list, str, bytes)):
                    note(f"{self.command} {parts.path} -> streaming")
                    self._send_stream(payload)
                    return
                note(f"{self.command} {parts.path} -> {status}")
                self._send_json(status, payload)
                return
            self._send_json(404, {"detail": f"no route for {self.command} {parts.path}"})

        do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = _dispatch

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
        # Almost always a previous run that is still holding the port. Said
        # plainly here, because the stock message is `[Errno 48] Address already
        # in use` and the next thing anyone does is guess.
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
