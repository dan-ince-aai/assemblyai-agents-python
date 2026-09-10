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

2. **Decide where each tool runs** before writing code (see the next section).

3. **Write the declaration** in one module (e.g. `agent.py`): `@tool` functions
   plus a module-level `agent = VoiceAgent(...)`. Nothing in it should touch the
   network at import time, so it can be imported by tests and by the server.

4. **Serve the backend** if any tool has `http=` or a pre-connect request is
   declared: one `POST /tools/{name}` route that authorises the request and
   calls `TOOLS[name].invoke(**arguments)`, one route per pre-connect URL, and
   a webhook route that calls `verify()` on the raw body. FastAPI is the natural
   fit but any framework works. Ask the user for the public HTTPS base URL (or
   suggest a tunnel for development) and read it from an environment variable.

5. **Deploy** with `client.agents.create(agent)`; print and persist the returned
   `id` (env var, `.agent_id` file, or the user's config). On later changes use
   `client.agents.update(agent_id, agent)`, which sends the whole declaration
   because the endpoint replaces the stored agent. Do not create a new agent on
   every run.

6. **Verify** in this order, cheapest first:
   - `python -c "from agent import agent; print(agent.to_request().model_dump(exclude_none=True))"`
     – a `ConfigurationError` here names the exact rule broken.
   - Unit-test the tool functions with `assemblyai_agents.testing`
     (`create_tool_context`, `get_tool`); tools are plain callables.
   - `client.agents.get(agent_id)` – confirm tools and pre-connect came back
     with the right URLs (header values are never echoed; that is expected).
   - **End to end through a tunnel, no mic or phone needed**: run the bundled
     `scripts/e2e_check.py` (see below). It is the fastest way to prove the
     platform actually reaches the backend, and its recorded requests show the
     exact body/headers the platform sends.
   - A real call: `AgentConnection` from a terminal (needs the `[audio]` extra
     and PortAudio), or a phone number.

7. **Attach a phone number** only after every tool has `http=`: the SDK refuses
   `assign_agent(..., agent=agent)` for a declaration with client-resident
   tools, because a phone call has no connected client to run them.

## Proving it works: `scripts/e2e_check.py`

The script in this skill's `scripts/` folder needs `ngrok` (configured with an
auth token) or `cloudflared` on `PATH`, plus `ASSEMBLYAI_API_KEY`. It opens a
tunnel to a local port, imports the declaration with `PUBLIC_BASE_URL` set to
the tunnel URL, deploys a throwaway copy of the agent, opens a WebSocket session
with no device audio, injects an utterance as a text turn, and records every
request the platform makes to the tool paths. Exit code 0 means the platform
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
- the utterance clearly needs the named tool; the model does not always act on
  a text turn, so the script retries with a fresh session (`--attempts`, default 3).

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

Default to `http=` tools for anything the user would ship. A handy pattern is a
`hosted(path)` helper that returns the HTTP config when `PUBLIC_BASE_URL` is set
and `None` otherwise, so the same declaration runs client-resident on a laptop
and hosted in production (see `examples/pizza_line.py` in the SDK repo).

## Writing a tool that the SDK accepts

`@tool` derives the JSON schema from the signature and refuses anything the
server would reject, at import time. The rules, and why they exist:

- **snake_case function name** – it *is* the tool name the model calls, so the
  two cannot drift. Avoid `aai_credit_card_luhn_check` and
  `aai_pre_connect_context`; those route to platform tools.
- **Docstring first paragraph** is the description the model reads to decide
  whether to call the tool; required. Put per-parameter text under `Args:`.
- **Every parameter typed**: `str`, `int`, `float`, `bool`, `list[T]`,
  `dict[str, T]`, `Literal[...]`, `Enum`, `Optional[T]`, pydantic `BaseModel`.
  A default makes it optional. No `*args`/`**kwargs`.
- **Return annotation required** and JSON-serialisable (`dict`, `list`, `str`,
  `int`, `float`, `bool`, `None`, `BaseModel`).
- `timeout_seconds` 1–300 (default 120). Set it low (5–15) for anything a caller
  waits through: past a few seconds they are listening to silence.
- `execution_mode` only `interactive` in v1. `response_instructions` adds static
  wording after success/error. `dtmf_collected_arguments` reads a parameter from
  the phone keypad (card numbers, account IDs).
- A parameter annotated `ToolContext` is injected, not part of the schema. Only
  use it when the tool is run through `Tool.invoke(context=...)` (your backend
  or the testing double); `AgentConnection` passes model arguments only.

Prompt guidance for `system_prompt`: spoken output, so short sentences, no
markdown, no lists; state when to call each tool by name; tell it what to do
when a lookup fails. `VoiceAgent` dedents the prompt, so indent freely.

## Backend contracts (what the platform sends you)

- **HTTP tool call.** `POST`/`PUT`/`PATCH` tools receive the model's arguments
  as the JSON body shaped by the tool's parameter schema; `GET`/`DELETE` tools
  receive them as query parameters. Whatever JSON you return is stringified for
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
| `AuthenticationError: Unauthorized` | Wrong or missing `ASSEMBLYAI_API_KEY`, or the key belongs to the other regional host. |
| `NotFoundError agent_not_found` on the WebSocket | Agent was created on the other host, or was deleted. |
| `DeviceAudioNotInstalledError` | `pip install "assemblyai-agents[audio] @ git+..."` after installing PortAudio. |
| Model never calls the tool | Sharpen the docstring's first paragraph and say in the system prompt when to call it. |

## Deliverable checklist

- `agent.py` with `@tool` functions and a module-level `VoiceAgent`, importable without side effects.
- A backend (when any tool is hosted): `/tools/{name}` with auth, pre-connect route(s), webhook route with `verify()`; secrets and the public base URL from environment variables; a short run/expose note.
- `deploy.py` that creates on first run and updates when an agent id is present.
- Tests for the tools using `assemblyai_agents.testing`, plus one asserting on `agent.to_request()`.
- A README snippet for the user: install line, env vars, how to deploy, how to try it.
- One `e2e_check.py` run that ends in `PASS`, with its recorded request pasted into the report.
