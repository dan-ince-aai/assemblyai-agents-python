"""Prove the platform can reach your tools from your laptop: no phone number, no mic.

(Bundled with the assemblyai-agents-sdk skill; identical to examples/e2e_check.py in the SDK repo.)

    pip install "git+https://github.com/dan-ince-aai/assemblyai-agents-python.git"
    brew install ngrok            # or cloudflared; either must be on PATH (ngrok needs `ngrok config add-authtoken`)
    export ASSEMBLYAI_API_KEY=...
    python examples/e2e_check.py --module pizza_line --path examples \
        --utterance "Hi, what's the status of order W004?" --tool lookup_order

What it does, in order:

1. Gets a public HTTPS address for your local port. Pass ``--public-url`` if
   you already have one, from a tunnel you run yourself or a staging host;
   otherwise, as a convenience, it starts ngrok or cloudflared for you. How
   your machine becomes reachable is your business, not the SDK's: nothing in
   ``assemblyai_agents`` knows what a tunnel is.
2. Sets ``PUBLIC_BASE_URL`` to that URL and imports your declaration module, so
   every ``http=`` tool URL and pre-connect URL points at the tunnel. Your
   declaration must build those URLs from ``PUBLIC_BASE_URL`` (see pizza_line.py).
3. Serves the tools itself on that port, at the exact paths your declaration
   uses, by calling ``Tool.invoke`` - or, with ``--forward http://127.0.0.1:8000``,
   proxies every request to your own running backend instead. Either way each
   request the platform makes is recorded.
4. Deploys a throwaway copy of the agent, opens a WebSocket session with no
   device audio, waits for the greeting to finish, then hands the utterance to
   the model through ``reply.create`` instructions. (A ``conversation.message``
   is not reliably seen by the model, and a reply requested while the greeting
   is still playing replaces the greeting, so neither is used.)
5. Waits for the platform to call a tool endpoint through the tunnel, lingers a
   few seconds so the follow-up requests land too, prints the recorded requests
   (method, path, arguments, headers with secrets masked), and exits 0 on
   success. The throwaway agent is deleted and the tunnel closed.

Requests to any other path are recorded as well, so this also shows a bring
your own LLM endpoint being called: with ``byo_llm_server.py`` running behind
``--forward``, the report contains the platform's ``POST /v1/chat/completions``,
the tool call your endpoint asked for, and the second completion carrying the
tool result.

The injected turn is a test convenience; the check retries with a fresh session
(``--attempts``) to absorb model variance. Export any other variable your
declaration reads (for example ``TOOL_SECRET``) alongside ``ASSEMBLYAI_API_KEY``.
ngrok's free tier allows one agent session at a time, so stop any other ngrok
first or pass ``--tunnel cloudflared``. Pre-connect requests are telephony-only
and are not exercised here.
"""

import argparse
import asyncio
import dataclasses
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qsl, urlsplit

import httpx

HEALTH_PATH = "/__e2e_health"
SKIP_WARNING = {"ngrok-skip-browser-warning": "1"}  # ngrok's free tier shows an interstitial to browsers
GREETING_TIMEOUT = 20.0  # seconds to wait for the platform's opening turn before injecting anyway


# --------------------------------------------------------------------------- tunnel


def start_tunnel(kind: str, port: int, log_path: str) -> tuple[subprocess.Popen, str]:
    if kind == "auto":
        kind = "ngrok" if shutil.which("ngrok") else "cloudflared" if shutil.which("cloudflared") else ""
        if not kind:
            sys.exit("no tunnel binary found: install ngrok (and run `ngrok config add-authtoken ...`) or cloudflared")
    log = open(log_path, "w")
    if kind == "ngrok":
        proc = subprocess.Popen(["ngrok", "http", str(port), "--log", "stdout", "--log-format", "json"], stdout=log, stderr=subprocess.STDOUT)
        for _ in range(40):
            time.sleep(1)
            try:
                tunnels = httpx.get("http://127.0.0.1:4040/api/tunnels", timeout=3).json().get("tunnels", [])
            except Exception:
                continue
            for t in tunnels:
                if t["public_url"].startswith("https://") and t.get("config", {}).get("addr", "").endswith(f":{port}"):
                    return proc, t["public_url"]
        proc.terminate()
        sys.exit("ngrok did not report a tunnel; log:\n" + open(log_path).read()[-1500:])
    if kind == "cloudflared":
        proc = subprocess.Popen(["cloudflared", "tunnel", "--url", f"http://127.0.0.1:{port}", "--no-autoupdate"], stdout=log, stderr=subprocess.STDOUT)
        for _ in range(40):
            time.sleep(1)
            m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", open(log_path).read())
            if m:
                return proc, m.group(0)
        proc.terminate()
        sys.exit("cloudflared did not print a trycloudflare.com URL; log:\n" + open(log_path).read()[-1500:])
    sys.exit(f"unknown tunnel kind {kind!r}")


def wait_reachable(public_url: str, seconds: int) -> None:
    deadline = time.time() + seconds
    last = None
    while time.time() < deadline:
        try:
            r = httpx.get(public_url + HEALTH_PATH, headers=SKIP_WARNING, timeout=10)
            if r.status_code == 200 and r.text.strip() == "ok":
                return
            last = f"HTTP {r.status_code}"
        except Exception as exc:
            last = type(exc).__name__
        time.sleep(2)
    sys.exit(
        f"tunnel {public_url} never answered the health probe ({last}). cloudflared quick tunnels "
        f"can take minutes to resolve, or fail on some networks; try --tunnel ngrok."
    )


# --------------------------------------------------------------------------- local server


@dataclasses.dataclass
class Recorded:
    at: float
    method: str
    path: str
    arguments: object
    headers: dict
    status: int
    elapsed_ms: float


class Recorder:
    def __init__(self) -> None:
        self.requests: list[Recorded] = []
        self._lock = threading.Lock()
        self.on_tool_call = None  # set by the async side: callable() -> None, thread-safe

    def add(self, item: Recorded, is_tool: bool) -> None:
        with self._lock:
            self.requests.append(item)
        if is_tool and self.on_tool_call is not None:
            self.on_tool_call()


def _mask(headers) -> dict:
    out = {}
    for key, value in headers.items():
        if key.lower() in ("authorization", "x-api-key", "cookie") and value:
            out[key] = value.split(" ")[0] + " ***" if " " in value else "***"
        else:
            out[key] = value
    return out


def _coerce(arguments: dict, schema: dict) -> dict:
    # Query-string values arrive as text; use the tool's own schema to restore numbers/booleans.
    props = (schema or {}).get("properties", {})
    out = {}
    for key, value in arguments.items():
        kind = props.get(key, {}).get("type")
        try:
            if kind == "integer":
                value = int(value)
            elif kind == "number":
                value = float(value)
            elif kind == "boolean":
                value = str(value).lower() in ("1", "true", "yes")
        except ValueError:
            pass
        out[key] = value
    return out


def make_handler(tools_by_path: dict, forward_base: str | None, recorder: Recorder, loop: asyncio.AbstractEventLoop):
    class Handler(BaseHTTPRequestHandler):
        server_version = "e2e-check"

        def log_message(self, *_):  # keep stdout for our own report
            pass

        def _respond(self, status: int, body: bytes, content_type: str = "application/json") -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _handle(self) -> None:
            started = time.time()
            parts = urlsplit(self.path)
            if parts.path == HEALTH_PATH:
                self._respond(200, b"ok", "text/plain")
                return
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b""
            if self.command in ("POST", "PUT", "PATCH"):
                try:
                    arguments = json.loads(raw) if raw else {}
                except ValueError:
                    arguments = {"__raw__": raw.decode(errors="replace")}
            else:
                arguments = dict(parse_qsl(parts.query))
            declared = tools_by_path.get(parts.path)

            if forward_base is not None:
                headers = {k: v for k, v in self.headers.items() if k.lower() not in ("host", "content-length")}
                try:
                    upstream = httpx.request(self.command, forward_base + self.path, content=raw, headers=headers, timeout=30)
                    status, body, ctype = upstream.status_code, upstream.content, upstream.headers.get("content-type", "application/json")
                except Exception as exc:
                    status, body, ctype = 502, json.dumps({"error": f"forward failed: {exc}"}).encode(), "application/json"
            elif declared is None:
                status, body, ctype = 404, json.dumps({"error": f"no tool served at {parts.path}"}).encode(), "application/json"
            else:
                if isinstance(arguments, dict) and self.command in ("GET", "DELETE"):
                    arguments = _coerce(arguments, declared.spec.parameters)
                try:
                    result = asyncio.run(declared.invoke(**arguments)) if isinstance(arguments, dict) else {"error": "arguments were not a JSON object"}
                    status, body = 200, json.dumps(result, default=str).encode()
                except Exception as exc:
                    status, body = 500, json.dumps({"error": f"{type(exc).__name__}: {exc}"}).encode()
                ctype = "application/json"
            self._respond(status, body, ctype)
            recorder.add(
                Recorded(started, self.command, parts.path, arguments, _mask(self.headers), status, (time.time() - started) * 1000),
                is_tool=declared is not None,
            )

        do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = _handle

    return Handler


# --------------------------------------------------------------------------- conversation


async def converse(agent_id: str, utterance: str, recorder: Recorder, wait_seconds: float, linger: float, attempt: int) -> bool:
    from assemblyai_agents import AgentConnection

    loop = asyncio.get_running_loop()
    reached = asyncio.Event()
    greeting_done = asyncio.Event()
    recorder.on_tool_call = lambda: loop.call_soon_threadsafe(reached.set)
    conn = AgentConnection(agent_id=agent_id, audio=False)
    transcripts = 0
    background: list[asyncio.Task] = []

    # The platform speaks the greeting as its own first reply. A reply.create sent
    # while that reply is in flight replaces the greeting and the injected text is
    # lost; and a conversation.message is not reliably visible to the model. So:
    # wait for the greeting's transcript, then carry the utterance in the
    # instructions of a reply.create.
    async def _inject():
        try:
            await asyncio.wait_for(greeting_done.wait(), timeout=GREETING_TIMEOUT)
        except asyncio.TimeoutError:
            print(f"  attempt {attempt}: no opening turn within {GREETING_TIMEOUT:.0f}s; injecting anyway")
        await asyncio.sleep(1.0)  # let the greeting's reply.done land
        print(f"  attempt {attempt}: handing the utterance to the model")
        await conn.session.create_reply(
            f'The caller just said: "{utterance}". Respond to the caller, calling your tools as needed.'
        )

    @conn.on_ready
    async def _ready(event):
        print(f"  attempt {attempt}: session {event.session_id} ready; waiting for the greeting")
        background.append(asyncio.create_task(_inject()))

    replied = asyncio.Event()

    @conn.on_agent_transcript
    def _agent(text):
        nonlocal transcripts
        transcripts += 1
        print(f"  attempt {attempt}: agent said: {text}")
        if transcripts == 1:
            greeting_done.set()
        else:
            replied.set()

    @conn.on_error
    def _error(event):
        print(f"  attempt {attempt}: session error {event.code.value}: {event.message}")
        reached.set()

    async with conn:
        run = asyncio.create_task(conn.run())
        try:
            await asyncio.wait_for(reached.wait(), timeout=wait_seconds)
        except asyncio.TimeoutError:
            print(f"  attempt {attempt}: no tool call reached the tunnel within {wait_seconds:.0f}s")
        else:
            # Stay on the line briefly: the turn that follows a tool call is
            # where the result is spoken, and where a BYO LLM endpoint is asked
            # for the wording.
            try:
                await asyncio.wait_for(replied.wait(), timeout=linger)
            except asyncio.TimeoutError:
                pass
        for task in background:
            task.cancel()
        await conn.aclose()
        try:
            await asyncio.wait_for(run, timeout=10)
        except Exception:
            pass
    recorder.on_tool_call = None
    return any(r.path in _TOOL_PATHS for r in recorder.requests)


_TOOL_PATHS: set = set()


# --------------------------------------------------------------------------- main


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("--module", required=True, help="module that defines `agent = VoiceAgent(...)`, e.g. pizza_line or myapp.agent")
    parser.add_argument("--path", default=".", help="directory to put on sys.path before importing --module (default: cwd)")
    parser.add_argument("--attr", default="agent", help="attribute holding the VoiceAgent (default: agent)")
    parser.add_argument("--utterance", required=True, help="what the caller says, phrased to make the model call a tool")
    parser.add_argument("--tool", help="name of the tool that must be reached (default: any HTTP tool)")
    parser.add_argument("--port", type=int, default=8788)
    parser.add_argument(
        "--public-url",
        metavar="URL",
        help="an HTTPS address that already reaches --port; skips starting a tunnel",
    )
    parser.add_argument(
        "--tunnel",
        choices=["auto", "ngrok", "cloudflared"],
        default="auto",
        help="which tunnel to start when --public-url is not given",
    )
    parser.add_argument("--forward", metavar="URL", help="proxy requests to your own running backend, e.g. http://127.0.0.1:8000, instead of serving the tools here")
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--wait", type=float, default=45, help="seconds to wait per attempt (greeting included) for the platform's tool call")
    parser.add_argument("--linger", type=float, default=8, help="seconds to stay connected after the tool call, to capture the reply turn")
    parser.add_argument("--reach-timeout", type=int, default=120, help="seconds to wait for the tunnel to answer")
    args = parser.parse_args()

    if not os.environ.get("ASSEMBLYAI_API_KEY"):
        sys.exit("set ASSEMBLYAI_API_KEY")
    log_path = os.path.join(os.environ.get("TMPDIR", "/tmp"), "e2e_check_tunnel.log")

    tunnel = None
    if args.public_url:
        public_url = args.public_url.rstrip("/")
        print(f"1. using the address you gave: {public_url}  ->  http://127.0.0.1:{args.port}")
    else:
        print("1. opening a tunnel (pass --public-url to use your own)")
        tunnel, public_url = start_tunnel(args.tunnel, args.port, log_path)
        print(f"   {public_url}  ->  http://127.0.0.1:{args.port}   (log: {log_path})")

    created = None
    httpd = None
    try:
        print("2. importing the declaration with PUBLIC_BASE_URL set")
        os.environ["PUBLIC_BASE_URL"] = public_url
        sys.path.insert(0, os.path.abspath(args.path))
        module = importlib.import_module(args.module)
        agent = getattr(module, args.attr)
        http_tools = [t for t in agent.tools or [] if t.spec.http is not None]
        resident = agent.client_resident_tool_names()
        if not http_tools:
            sys.exit("   the declaration has no tools with http=; build tool URLs from PUBLIC_BASE_URL (see pizza_line.hosted)")
        if resident:
            print(f"   note: client-resident tools {resident} are not exercised (they need a connected client)")
        stray = [t.name for t in http_tools if not t.spec.http.url.startswith(public_url)]
        if stray:
            sys.exit(f"   tools {stray} do not point at the tunnel; the declaration must read PUBLIC_BASE_URL at import time")
        tools_by_path = {urlsplit(t.spec.http.url).path: t for t in http_tools}
        _TOOL_PATHS.update(tools_by_path)
        for path, t in tools_by_path.items():
            print(f"   {t.spec.http.http_method.value:6s} {public_url}{path}  ->  {t.name}()")
        if args.tool and args.tool not in {t.name for t in http_tools}:
            sys.exit(f"   --tool {args.tool!r} is not an HTTP tool on this agent")

        print(f"3. serving tools on :{args.port}" + (f", forwarding to {args.forward}" if args.forward else " by calling Tool.invoke"))
        recorder = Recorder()
        loop = asyncio.new_event_loop()
        httpd = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(tools_by_path, args.forward, recorder, loop))
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        wait_reachable(public_url, args.reach_timeout)
        print("   tunnel answers the health probe")

        print("4. deploying a throwaway copy of the agent")
        from assemblyai_agents import Client
        client = Client()
        created = client.agents.create(dataclasses.replace(agent, name=f"{agent.name} (e2e check)"))
        print(f"   {created.id}")

        print("5. talking to it")
        success = False
        for attempt in range(1, args.attempts + 1):
            asyncio.run(converse(created.id, args.utterance, recorder, args.wait, args.linger, attempt))
            time.sleep(1)
            hits = [r for r in recorder.requests if r.path in tools_by_path and (args.tool is None or tools_by_path[r.path].name == args.tool)]
            if hits:
                success = True
                break
        print()
        print("requests the platform made through the tunnel:")
        if not recorder.requests:
            print("  (none)")
        for r in recorder.requests:
            label = f"tool={tools_by_path[r.path].name}" if r.path in tools_by_path else "not a tool path"
            print(f"  {r.method} {r.path}  {label}  status={r.status}  {r.elapsed_ms:.0f} ms")
            rendered = json.dumps(r.arguments)
            print(f"      arguments: {rendered if len(rendered) <= 600 else rendered[:600] + '…'}")
            print(f"      headers:   {json.dumps({k: v for k, v in r.headers.items() if k.lower() in ('authorization', 'content-type', 'user-agent')})}")
        print()
        if success:
            print(f"PASS: the platform called {args.tool or 'an HTTP tool'} on your backend through the tunnel.")
            return 0
        print("FAIL: no matching tool call reached the tunnel. Check the utterance clearly needs the tool, the tool's docstring, and the system prompt.")
        return 1
    finally:
        if created is not None:
            try:
                Client().agents.delete(created.id)
                print(f"deleted {created.id}")
            except Exception as exc:
                print(f"could not delete {created.id}: {exc}")
        if httpd is not None:
            httpd.shutdown()
        if tunnel is not None:
            tunnel.terminate()
            try:
                tunnel.wait(5)
            except Exception:
                tunnel.kill()


if __name__ == "__main__":
    sys.exit(main())
