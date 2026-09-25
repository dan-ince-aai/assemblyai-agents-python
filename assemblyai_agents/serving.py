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

    ANY  /tools/{name}                each @tool on the declaration
    POST <prefix>/chat/completions    `reply(turn)`, streamed as the platform wants
    ANY  <your pre-connect path>      a handler you pass in
    POST /webhooks/voice-agents       verified against your secret
    GET  /healthz                     what is loaded

A tool and a pre-connect request arrive with whatever method you declared for
them: GET and DELETE carry the arguments as a query string, the others as a
JSON body. The chat path is whatever path your `llm.base_url` has, followed by
`/chat/completions`, so `https://host/v1` and `https://host` both work.

For anything you would rather host yourself, `routes()` hands back the same
handlers as plain callables to mount wherever you like.
"""

import asyncio
import json
import re
import socket
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Mapping, Optional, get_type_hints
from urllib.parse import parse_qsl, urlsplit

from pydantic import BaseModel, TypeAdapter, ValidationError

from . import webhooks as _webhooks
from ._context import ToolContext
from ._exceptions import WebhookVerificationError
from .byo import Turn, json_body, stream

_JSON = "application/json"
_TEXT = "text/plain; charset=utf-8"
_ANY_METHOD = ("GET", "POST", "PUT", "PATCH", "DELETE")

# The platform fails a pre-connect request whose response is larger than this,
# which refuses the call outright on an entry set to reject.
PRE_CONNECT_RESPONSE_LIMIT_BYTES = 8 * 1024

# A tool that raises is answered with this closed shape, the same one the
# hosted runtime uses. The platform hands a tool's error body to the model as it
# was sent, so an exception message here would be read out to the caller, and
# an exception message is where a URL or a key lands by accident.
_TOOL_RAISED = "tool_raised"


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


class _Text(str):
    """A result to send as plain text rather than as a JSON string."""


def _coerce(declared: Any, arguments: dict) -> dict:
    """Arguments converted to the handler's annotations.

    A GET or DELETE tool's query string is all strings, and a nested object
    arrives as a plain dict. The hosted runtime validates both against the
    handler's type hints, so a tool behaves the same served from here. As
    there, an argument the handler does not declare is dropped.
    """
    hints = get_type_hints(declared.spec.target)
    coerced = {}
    for name, value in arguments.items():
        annotation = hints.get(name)
        if annotation is None or annotation is ToolContext:
            continue
        coerced[name] = TypeAdapter(annotation).validate_python(value)
    return coerced


def _arguments(query: str, body: bytes) -> Any:
    if body:
        return json.loads(body)
    return dict(parse_qsl(query))


def _render(result: Any) -> Any:
    if isinstance(result, str):
        # Unquoted, because the platform hands the body to the model as text
        # and a JSON string literal would reach it wrapped in quotes.
        return _Text(result)
    if isinstance(result, BaseModel):
        return result.model_dump(mode="json")
    return result


def routes(
    agent: Any,
    *,
    reply: Optional[Callable[[Turn], Any]] = None,
    tool_secret: Optional[str] = None,
    llm_key: Optional[str] = None,
    pre_connect: Optional[Mapping[str, Callable]] = None,
    webhook_secret: Optional[str] = None,
    on_event: Optional[Callable[[dict], None]] = None,
    context: Optional[ToolContext] = None,
    log: Optional[Callable[[str], None]] = _print,
) -> dict:
    """The handlers, without a server, for mounting in your own framework.

    Keyed by `(method, compiled path pattern)`. Each takes
    `(path, query, body_bytes, headers)` and returns `(status, payload)`, where
    a payload that is an iterator is streamed as Server-Sent Events and a `str`
    payload is plain text. `headers` must look names up case-insensitively.
    """
    tools = {declared.name: declared for declared in agent.tools or []}

    def note(message: str) -> None:
        if log is not None:
            log(message)

    def check(headers: Mapping[str, str], secret: Optional[str], status: int = 401) -> None:
        if secret and headers.get("Authorization") != f"Bearer {secret}":
            raise Refused(status, "unauthorized")

    def run_tool(path: str, query: str, body: bytes, headers: Mapping[str, str]):
        check(headers, tool_secret)
        name = path.rsplit("/", 1)[-1]
        declared = tools.get(name)
        if declared is None:
            raise Refused(404, f"no tool named {name!r}")
        arguments = _arguments(query, body)
        if not isinstance(arguments, dict):
            raise Refused(422, "arguments were not a JSON object")
        try:
            arguments = _coerce(declared, arguments)
        except ValidationError as exc:
            raise Refused(422, str(exc)) from exc
        try:
            result = asyncio.run(declared.invoke(context=context, **arguments))
        except Exception as exc:
            # The message stays in your own log; the platform only hears the type.
            note(f"tool {name!r} raised {type(exc).__name__}: {exc}")
            return 500, {"error": _TOOL_RAISED, "type": type(exc).__name__}
        return 200, _render(result)

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
        except WebhookVerificationError as exc:
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

    def lookup(handler: Callable) -> Callable:
        def run(path: str, query: str, body: bytes, headers: Mapping[str, str]):
            check(headers, tool_secret)
            result = handler(_arguments(query, body))
            if asyncio.iscoroutine(result):
                result = asyncio.run(result)
            # An empty body is not JSON, and the platform fails the entry on it.
            result = {} if result is None else _render(result)
            size = len(json.dumps(result, default=str).encode())
            if size > PRE_CONNECT_RESPONSE_LIMIT_BYTES:
                note(
                    f"pre-connect {path} answered {size} bytes; the platform fails "
                    f"any response over {PRE_CONNECT_RESPONSE_LIMIT_BYTES}"
                )
            return 200, result

        return run

    table: dict = {}
    # The platform sends a tool's arguments with the tool's own method.
    for method in _ANY_METHOD:
        table[(method, re.compile(r"^/tools/[^/]+$"))] = run_tool
    # The OpenAI client posts to `<base_url>/chat/completions`, so the prefix is
    # whatever path the agent's `llm.base_url` has, including none.
    table[("POST", re.compile(r"^(/.*)?/chat/completions$"))] = replies
    table[("POST", re.compile(r"^/webhooks/voice-agents$"))] = webhook
    table[("GET", re.compile(r"^/healthz$"))] = health

    for path, handler in (pre_connect or {}).items():
        run = lookup(handler)
        for method in _ANY_METHOD:
            table[(method, re.compile(rf"^{re.escape(path)}$"))] = run

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
    context: Optional[ToolContext] = None,
    log: Optional[Callable[[str], None]] = _print,
    background: bool = False,
) -> Any:
    """Answer the platform's HTTPS requests with your own functions.

    `context` is handed to any tool that takes a `ToolContext`.

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
        context=context,
        log=log,
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
                    self._send(refused.status, {"detail": refused.detail})
                    return
                except Exception as exc:
                    # Anything else that raised: log it in full, keep serving,
                    # and answer with the type only, since the platform can hand
                    # this body to the model.
                    note(f"{self.command} {parts.path} -> 500 {type(exc).__name__}: {exc}")
                    self._send(500, {"detail": type(exc).__name__})
                    return
                if isinstance(payload, Iterator):
                    note(f"{self.command} {parts.path} -> streaming")
                    self._send_stream(payload)
                    return
                note(f"{self.command} {parts.path} -> {status}")
                self._send(status, payload)
                return
            self._send(404, {"detail": f"no route for {self.command} {parts.path}"})

        do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = _dispatch

        def _send(self, status: int, payload: Any) -> None:
            # A `str` result goes out as plain text, and JSON `null` is still a
            # body; only a 204 is sent empty.
            if status == 204:
                encoded, content_type = b"", None
            elif isinstance(payload, _Text):
                encoded, content_type = str(payload).encode(), _TEXT
            else:
                encoded, content_type = json.dumps(payload, default=str).encode(), _JSON
            self.send_response(status)
            if content_type:
                self.send_header("Content-Type", content_type)
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
    served = ["/tools/{name}", "<prefix>/chat/completions", "/webhooks/voice-agents", "/healthz"]
    served += sorted(pre_connect or {})
    note(f"serving {agent.name!r} on http://{host}:{server.server_address[1]} — {', '.join(served)}")
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
