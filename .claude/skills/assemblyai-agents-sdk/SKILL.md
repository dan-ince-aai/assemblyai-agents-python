---
name: assemblyai-agents-sdk
description: Build, deploy and operate AssemblyAI voice agents in Python with the assemblyai-agents SDK, the backend SDK for the Voice Agents API. Use this whenever the user wants a Python voice agent, phone agent, IVR, receptionist, order-line or any call-handling bot; wants to add, host or test tools for one; needs pre-connect (caller lookup) requests, Voice Agents webhooks, phone numbers, outbound calls or human transfers; or mentions assemblyai_agents, VoiceAgent, @tool, AgentConnection or "agents.assemblyai.com". Trigger even when the user only describes the agent's job in plain language ("make a bot that takes reservations") and never names the SDK.
---

# Building voice agents with `assemblyai-agents`

The SDK is a **backend** SDK: you declare the agent and its tools in Python,
deploy the declaration over REST, and serve the tool logic, pre-connect lookups
and webhook handling from the user's own backend. The platform owns the call
(speech-to-text, LLM, text-to-speech, turn-taking, telephony); it reaches the
backend over HTTPS when the model calls a tool. Keep that split in mind: the
deliverable is usually a small service plus a declaration, not a client app.

`references/sdk-reference.md` is the full API surface (every class, method,
field and exception). Read it when you need a signature or an exact field name;
this file covers the workflow and the decisions.

## Workflow

1. **Install and check credentials.**
   ```bash
   pip install "git+https://github.com/dan-ince-aai/assemblyai-agents-python.git"
   python -c "import assemblyai_agents; print(assemblyai_agents.__version__)"
   ```
   The client reads `ASSEMBLYAI_API_KEY`. If it is not set, ask the user for it
   (or where it lives) rather than guessing; never write a key into source.
   Default host is `https://agents.assemblyai.com`; `https://agents.us.assemblyai.com`
   is the US deployment. Resources live per host, so pick one and stay on it.
   Agent ids are disjoint across hosts and the same *names* exist on both, so
   persist the id and never look an agent up by name.

2. **Decide where each tool runs** before writing code (see the next section).

3. **Write the declaration** in one module (e.g. `agent.py`): `@tool` functions
   plus a module-level `agent = VoiceAgent(...)`. Nothing in it should touch the
   network at import time, so it can be imported by tests and by the server.

4. **Serve the backend** if any tool has `http=` or a pre-connect request is
   declared: one `POST /tools/{name}` route that authorises the request and
   calls `TOOLS[name].invoke(**arguments)`, one route per pre-connect URL, and
   a webhook route that calls `verify()` on the raw body. FastAPI is the natural
   fit but any framework works. Ask the user for the public HTTPS base URL and
   read it from an environment variable (`PUBLIC_BASE_URL`). The API checks at
   create/update time that every tool and pre-connect hostname resolves in
   public DNS (`ValidationError: … URL host … does not resolve`), so a
   placeholder cannot be deployed: during development expose the local server
   with `ngrok http 8000` or `cloudflared tunnel --url http://127.0.0.1:8000`
   (or let `scripts/e2e_check.py` do it) and keep the server running — a dead
   origin shows up only as the model apologising to the caller.

5. **Deploy** with `client.agents.create(agent)`; print and persist the returned
   `id` (env var, `.agent_id` file, or the user's config). On later changes use
   `client.agents.update(agent_id, agent)`, which sends the whole declaration
   because the endpoint replaces the stored agent. Do not create a new agent on
   every run. If `update()` raises `NotFoundError`, the stored agent is gone:
   create again and overwrite the stored id.

6. **Verify** in this order, cheapest first:
   - `python -c "from agent import agent; r = agent.to_request(); print([t.name for t in r.tools or []], [p.http.url for p in r.pre_connect_requests or []])"`
     – a `ConfigurationError` here names the exact rule broken. (A full
     `model_dump()` also prints the tool header secrets, so avoid it in logs.)
   - Unit-test the tool functions with `assemblyai_agents.testing`
     (`create_tool_context`, `get_tool`); tools are plain callables.
   - `client.agents.get(agent_id)` – confirm tools and pre-connect came back
     with the right URLs (header values are never echoed; that is expected).
   - **End to end through a tunnel, no mic or phone needed**: run the bundled
     `scripts/e2e_check.py` (see below). It is the fastest way to prove the
     platform actually reaches the backend, and its recorded requests show the
     exact body/headers the platform sends.
   - Mic-less spot check: `AgentConnection(agent_id=..., audio=False)`; wait
     for the first `on_agent_transcript` (the greeting), then
     `await conn.session.create_reply('The caller just said: "…". Respond to the caller, calling your tools as needed.')`.
     Do not rely on `say()` / `conversation.message`: in testing the model did
     not see its content, and a `reply.create` sent while the greeting is still
     playing replaces the greeting. `scripts/e2e_check.py` does exactly this.
   - A real call: `AgentConnection` from a terminal (needs the `[audio]` extra
     and PortAudio), or a phone number.
   - When a hosted tool fails, the caller only hears an apology and the client
     sees no error: read the server/tunnel logs and
     `client.sessions.get(session_id).artifacts` (the `timeline` artifact lists
     each turn with its `trigger`, e.g. `tool_result`).

7. **Attach a phone number** only after every tool has `http=`: the SDK refuses
   `assign_agent(..., agent=agent)` for a declaration with client-resident
   tools, because a phone call has no connected client to run them.

## Proving it works: `scripts/e2e_check.py`

The script in this skill's `scripts/` folder needs `ngrok` (configured with an
auth token) or `cloudflared` on `PATH`, plus `ASSEMBLYAI_API_KEY`. It opens a
tunnel to a local port, imports the declaration with `PUBLIC_BASE_URL` set to
the tunnel URL, deploys a throwaway copy of the agent, opens a WebSocket session
with no device audio, waits for the greeting, hands the utterance to the model
through `reply.create` instructions, and records every request the platform
makes to the tool paths. Exit code 0 means the platform
called the tool through the tunnel. The throwaway agent is deleted afterwards.

```bash
python <skill-dir>/scripts/e2e_check.py --module agent --path . \
    --utterance "I'd like to book a cleaning next Tuesday morning" --tool check_availability
# add --forward http://127.0.0.1:8000 to route the platform's calls to the user's running server
```

Requirements it imposes on the declaration, so design for them from the start:
- the module exposes `agent = VoiceAgent(...)` (or pass `--attr`) and has no
  network side effects at import;
- tool URLs are built from `os.environ["PUBLIC_BASE_URL"]` (a `hosted(path)`
  helper), so pointing them at a tunnel is a matter of setting one variable;
- the utterance clearly needs the named tool; the script retries with a fresh
  session (`--attempts`, default 3) to absorb model variance;
- `ASSEMBLYAI_API_KEY` and any other variable the declaration reads (for
  example `TOOL_SECRET`) are exported; the script sets only `PUBLIC_BASE_URL`;
- no other ngrok session is running (the free tier allows one; stop the one
  behind a dev deployment first, or pass `--tunnel cloudflared`).

Pre-connect is telephony-only and is not exercised by this check. ngrok's free
tier serves an interstitial to browsers; the script sends the
`ngrok-skip-browser-warning` header on its own probes and the platform is
unaffected. If `cloudflared` never becomes reachable (its quick-tunnel DNS can
lag for minutes on some networks) use `--tunnel ngrok`.

## Where a tool runs

| Tool declared… | Who runs it | Works on | Use when |
| --- | --- | --- | --- |
| with `http=PlaintextHttpToolConfig(url=..., http_method=..., headers=[...])` | the user's backend; platform POSTs/GETs the arguments | phone and WebSocket | production, anything that touches the user's data |
| without `http=` (client-resident) | the process holding the WebSocket, via `AgentConnection(tools={name: fn})` | WebSocket only | local development, desktop/browser sessions needing local state |

`PlaintextHttpToolConfig`, `HttpToolHeaderInput`, `HttpMethod`,
`ResponseInstructions`, `DtmfCollectionProfile`, `ExecutionMode`,
`LlmConfigRequest` and the other request models are **not** top-level exports:

```python
from assemblyai_agents import VoiceAgent, tool, ToolContext, Captured, Header, PreConnectRequest, HumanTransfer
from assemblyai_agents.models.rest import HttpMethod, HttpToolHeaderInput, PlaintextHttpToolConfig, ResponseInstructions
```

Always pass `http_method=HttpMethod.POST` (or `GET`) explicitly: `HttpMethod` is
a plain `Enum`, and the field's default is the bare string `"POST"`, which
serialises with a pydantic warning and compares unequal to the enum.

Default to `http=` tools for anything the user would ship. A handy pattern is a
`hosted(path)` helper that returns the HTTP config when `PUBLIC_BASE_URL` is set
and `None` otherwise, so the same declaration runs client-resident on a laptop
and hosted in production, and `e2e_check.py` can point it at a tunnel:

```python
import os
from assemblyai_agents import VoiceAgent, tool
from assemblyai_agents.models.rest import HttpMethod, HttpToolHeaderInput, PlaintextHttpToolConfig

PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "").rstrip("/")
TOOL_SECRET = os.environ.get("TOOL_SECRET", "change-me")

def hosted(path: str) -> PlaintextHttpToolConfig | None:
    """HTTP config pointing the platform at your backend, or None to run the tool in-process."""
    if not PUBLIC_BASE_URL:
        return None
    return PlaintextHttpToolConfig(
        url=f"{PUBLIC_BASE_URL}{path}",
        http_method=HttpMethod.POST,
        headers=[HttpToolHeaderInput(name="Authorization", value=f"Bearer {TOOL_SECRET}")],
    )

@tool(timeout_seconds=10, http=hosted("/tools/lookup_order"))
async def lookup_order(order_id: str) -> dict:
    """Look up the status of a customer's order by its order number.

    Args:
        order_id: The order number the caller read out, like W004.
    """
    ...
```

## Writing a tool that the SDK accepts

`@tool` derives the JSON schema from the signature and refuses anything the
server would reject, at import time. The rules, and why they exist:

- **snake_case function name** – it *is* the tool name the model calls, so the
  two cannot drift. Avoid `aai_credit_card_luhn_check` and
  `aai_pre_connect_context`; those route to platform tools.
- **Docstring first paragraph** is the description the model reads to decide
  whether to call the tool; required. Put per-parameter text under `Args:`.
- **Every parameter typed**: `str`, `int`, `float`, `bool`, `list[T]`,
  `dict[str, T]`, `Literal[...]`, `Enum`, `Optional[T]` / `T | None`, pydantic
  `BaseModel`. A default makes it optional. No `*args`/`**kwargs`.
- **Return annotation required** and JSON-serialisable (`dict`, `list`, `str`,
  `int`, `float`, `bool`, `None`, `BaseModel`).
- `timeout_seconds` 1–300 (default 120). Set it low (5–15) for anything a caller
  waits through: past a few seconds they are listening to silence.
- `execution_mode` only `interactive` in v1. `response_instructions` adds static
  wording after success/error. `dtmf_collected_arguments` reads a parameter from
  the phone keypad (card numbers, account IDs).
- A parameter annotated `ToolContext` is injected, not part of the schema, and
  only when the caller passes `context=` to `Tool.invoke`; `invoke(**arguments)`
  without it raises `TypeError`. The SDK ships no production context object
  (only the testing double), so for hosted tools either leave `ToolContext` out,
  declare it as `ctx: ToolContext = None` (accepted; `invoke(**arguments)` then
  runs with `ctx` unset — `Optional[ToolContext]` is refused), or build your own
  object satisfying the protocol in `/tools/{name}`. `AgentConnection` passes
  model arguments only.

Prompt guidance for `system_prompt`: spoken output, so short sentences, no
markdown, no lists; state when to call each tool by name; tell it what to do
when a lookup fails. `VoiceAgent` dedents the prompt, so indent freely.

## Backend contracts (what the platform sends you)

- **HTTP tool call.** `POST`/`PUT`/`PATCH` tools receive the model's arguments
  as the JSON body shaped by the tool's parameter schema; `GET`/`DELETE` tools
  receive them as query parameters, i.e. as strings — `Tool.invoke` does not
  coerce, so restore numbers/booleans from `tool.spec.parameters` first. Whatever JSON you return is stringified for
  the model. The headers configured on the tool are sent verbatim, so put a
  shared secret in an `Authorization` header and check it.
- **Pre-connect request.** Telephony only: called before a phone call is
  answered (never on a WebSocket session), within the entry's timeout (800 ms ceiling; `timeout_ms` only lowers it). Return JSON;
  `returns=[Captured(name, path, default)]` pulls values out of it (dotted
  paths, e.g. `customer.tier`). With `allow_overrides=True` a top-level
  `"greeting"` replaces the greeting for that call; a top-level `"reject": true`
  aborts the call. Everything else fails open. At most two entries; a later
  entry's `sends` may only name earlier captures. Captured values reach the
  model via the `aai_pre_connect_context` platform tool.
- **Webhook delivery.** `POST` with an `X-AAI-Signature` header. Call
  `verify(raw_body_bytes, header, secret)` **before** parsing JSON; it returns
  the event dict or raises `WebhookVerificationError`. Events:
  `session.started`, `session.completed`, `call.connected`, `call.ended`,
  `call.failed`.

## Telephony

- Managed number: `client.phone_numbers.purchase_available(PurchaseAvailablePhoneNumberRequest(country_code="US", number_type=NumberType.local, area_code=415, agent_id=...))`. Billable: confirm with the user before running it.
- Own number: `import_(ImportPhoneNumberRequest(phone_number, termination_uri))` then `assign_agent(number, PhoneNumberAssignAgentRequest(agent_id=...), agent=agent)`.
- Outbound: `client.calls.create(CreateCallRequest(from_number, to_number))`; `from_number` must be on the account with an agent assigned.
- Human transfer: `transfer_targets=[HumanTransfer(name, phone_number, mode="cold"|"warm", ...)]` **requires** `outbound_trunk_id`. Consult fields are warm-only. E.164 everywhere (`+14155550123`).
- Telephony audio is `audio/pcmu`/`audio/pcma`; the default `audio/pcm` at 24 kHz is for WebSocket clients. Transfer targets, pre-connect and DTMF do nothing on a WebSocket session.

## Pitfalls the SDK will tell you about (and how to fix them)

| Symptom | Fix |
| --- | --- |
| `ConfigurationError: tool ... no description` | Add a docstring; its first paragraph is the description. |
| `... no type hint` / `type ... is not supported` | Annotate every parameter with a supported type; replace `Any`/bare `list` with concrete types. |
| `... no return annotation` | Add `-> dict` (or another JSON type). |
| `tool ... is listed twice` | Tool names must be unique within an agent. |
| `... holds client-resident tools ... only a WebSocket session can answer` | Give the tool an `http=` config before attaching a phone number. |
| `transfer targets ... need an outbound_trunk_id` | Set `outbound_trunk_id` on the agent. |
| `pre-connect url=... is not https` | Pre-connect and tool URLs must be `https://`. |
| `ValidationError: … URL host … does not resolve` on create/update | Tool and pre-connect hostnames must resolve in public DNS: use a tunnel URL, not a placeholder. |
| `AuthenticationError: Unauthorized` | Wrong or missing `ASSEMBLYAI_API_KEY`, or the key belongs to the other regional host. |
| `on_error` receives `SessionError(code=agent_not_found)`, then the next send raises `ConnectionClosedError` 1008 | The agent lives on the other regional host or was deleted; `async with conn:` itself does not raise. Redeploy and store the new id. |
| `say()` produces silence, or the agent answers as if nothing was said | Text turns are not reliably visible to the model. To stand in for a caller in a test, wait for the greeting and then `create_reply(instructions='The caller just said: "…"…')`. |
| The agent opened with a generic "Hello, how can I help?" instead of the configured greeting | A `reply.create` was sent while the greeting was still playing and replaced it. Inject only after the first agent transcript. |
| The agent apologises that it cannot access the system | Your tool endpoint was unreachable, slow, or non-2xx. Check the tunnel/server logs and the session's `timeline` artifact. |
| `DeviceAudioNotInstalledError` | `pip install "assemblyai-agents[audio] @ git+..."` after installing PortAudio. |
| Model never calls the tool | Sharpen the docstring's first paragraph and say in the system prompt when to call it. |

## Deliverable checklist

- `agent.py` with `@tool` functions and a module-level `VoiceAgent`, importable without side effects.
- A backend (when any tool is hosted): `/tools/{name}` with auth, pre-connect route(s), webhook route with `verify()`; secrets and the public base URL from environment variables; a short run/expose note.
- `deploy.py` that creates on first run and updates when an agent id is present.
- Tests for the tools using `assemblyai_agents.testing`, plus one asserting on `agent.to_request()`.
- A README snippet for the user: install line, env vars, how to deploy, how to try it.
- One `e2e_check.py` run that ends in `PASS`, with its recorded request pasted into the report.
