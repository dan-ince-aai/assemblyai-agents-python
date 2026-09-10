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

## Start from the starter

Unless the user has an existing project, copy `examples/starter/` out of the
SDK repo and work in that. It is a running agent with the whole loop already
wired: a mocked system of record, tools, a staged reply engine, a local
rehearsal harness, whole-call tests, a scripted driver against the deployed
agent, and deploy. Renaming it and rewriting four files is faster and far less
error-prone than assembling the same thing from scratch, and it starts with
every platform trap below already handled.

```bash
cp -r <sdk-repo>/examples/starter my-agent && cd my-agent
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python \
    "git+https://github.com/dan-ince-aai/assemblyai-agents-python.git" \
    fastapi "uvicorn[standard]" pytest pytest-asyncio httpx
.venv/bin/python rehearse.py happy      # a whole call, offline, in milliseconds
.venv/bin/python -m pytest -q
```

Then change, in this order: `store.py` (their systems), `agent.py` (their tools
and prompt), `reply.py` (their call flow), `tests/test_call.py` (a test per
call worth caring about). Read its README for what each file is for.

Build from nothing only when the user asks for something the starter's shape
does not fit, and even then read `reply.py` and `agent.py` first for the
conventions.

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

## Bring your own LLM

`llm=LlmConfigRequest(base_url, model, api_key)` moves response generation to
any OpenAI-compatible chat-completions endpoint, so the user's backend can own
the words as well as the tools. Reach for it when they ask to use their own
model, an open-weights model they host, a gateway, or deterministic logic
instead of a model. The platform still owns speech, turn-taking and telephony.
`base_url` must be HTTPS and is DNS-checked at create/update like tool URLs;
`api_key` is write-only.

The contract, captured from a live session (build against this, not the OpenAI
docs alone):

- `POST {base_url}/chat/completions` with `Authorization: Bearer <api_key>`, and
  a 10 second read timeout — get the first chunk out fast, do slow work in tools.
- `stream: true` on every call with `stream_options: {"include_usage": true}`,
  so the endpoint **must** answer with Server-Sent Events in OpenAI's
  `chat.completion.chunk` shape. A plain JSON body will not do.
- `messages[0]` is the agent's `system_prompt` plus the platform's own
  spoken-output guidance; the greeting arrives as an `assistant` message.
- `tools` is OpenAI function form with `tool_choice: "auto"`, but with a second
  `type: "function"` and the platform's `timeout_seconds`/`execution_mode`
  nested inside `function` — read the name from `tool["function"]["name"]`.
- Emitting `tool_calls` makes the platform run that tool and call the endpoint
  again with a `tool` message (plus `tool_call_id`), then a `system` note like
  "The function call … has just completed". The tool message is therefore
  usually not the last one: the cue to speak is **a tool result with no
  assistant text after it**. Answer a repeated call from the transcript rather
  than re-issuing it, or the caller waits through the same round trip twice;
  three consecutive failures and the platform tells you to stop.
- **The platform refuses a tool call carrying a value the call never
  established**, and this is the single most common way a BYO LLM build fails.
  It does not run the tool; it returns a `tool` message saying so and a system
  note: "The call has not established a value for `account_ref`, and the caller
  would not know it by heart … Do not send a value for it until that has
  returned one. Never invent a value." An empty string counts as invented, so
  **omit an unknown optional argument entirely** rather than sending `""`. Pass
  values as the caller said them, or exactly as an earlier tool result returned
  them.
- **Pre-connect captures do reach the endpoint.** On a phone call the platform
  runs the lookup itself and puts the result at the top of the transcript as an
  `aai_pre_connect_context` tool result:
  `{"variables": {"account_ref": "100200300412", "consumer_first_name": "Maria"}}`.
  Read the values from there; they count as established, so they can be passed
  on to other tools.
- **A tool message is not always JSON.** A refused or failed call arrives as
  prose with bracketed coaching text appended, so parse defensively and do not
  report a refusal to the caller as a failure of the thing the tool does. A
  keypad collection that ends early reads "The caller did not finish entering
  'card_number' on their keypad (too_short), so the tool was not called", which
  is not a declined card.
- **Keypad-collected parameters are hidden from the endpoint.** The platform
  strips them out of the tool schema it shows you and sets that tool's
  `execution_mode` to `hold` itself, because it does the collecting. Send only
  the arguments that remain.
- **`conversation.message` with role user is visible to your endpoint**, because
  you read the raw transcript. That makes `session.send_message(text)` followed
  by `create_reply()` the way to drive a multi-turn test of a BYO LLM agent, and
  those turns persist, unlike an instruction passed to `create_reply`.

### The `byo` module does the plumbing

`assemblyai_agents.byo` is the contract in this section, already written. Use
it rather than hand-rolling SSE and transcript parsing:

```python
from assemblyai_agents.byo import Memo, Responder, mount_fastapi

responder = Responder()

@responder.stage("identify", until=lambda turn: turn.result_of("verify_caller"))
def identify(turn):
    if not turn.caller_said:
        return turn.say("Could you give me your full name?")
    return turn.call("verify_caller", caller_said=turn.caller_said)

@responder.stage("close")
def close(turn):
    return turn.silence()

mount_fastapi(app, responder, tools=TOOLS, tool_secret=..., llm_key=...,
              pre_connect={"/pre-connect/lookup": lookup}, webhook_secret=...)
```

What it gives you, each of which is a trap from the list above:

| | |
| --- | --- |
| `Turn.from_request(body)` | the request, read: `caller_said`, `spoken`, `tools`, `preconnect`, `pending`, `results` |
| `turn.pending` | the tool result nothing has been said about yet, which is the cue to speak; `None` once something has |
| `turn.pending.ran` | `False` when the platform refused or failed the call, so a refusal is never reported as a result |
| `turn.preconnect` | the pre-connect captures, read out of the `aai_pre_connect_context` result |
| `turn.result_of(name)` | an earlier result, to read back rather than call again |
| `turn.answer_following(fragment)` | the caller's reply to a question you asked, for a value collected over turns |
| `turn.call(name, **args)` | drops arguments the call has not established, so the platform accepts it |
| `turn.say(text)` / `turn.silence()` | words, or nothing, which is how a finished call ends |
| `Responder` / `.stage(name, until=...)` | ordered stages, each handing over when its own test passes |
| `responder.respond(body)` | a `Reply` with `.stream()`, `.json()`, `.spoken`, `.tool` |
| `mount_fastapi(...)` | every route the platform calls, with the auth checks |
| `Memo` | a note of what has been said, for lines an interrupted turn would otherwise repeat |
| `digits_said(text)` | digits out of "four four seven one", "forty one eleven", "double one" |
| `tool_runner(TOOLS)` | run a tool by name, with a bad name and bad arguments told apart |

`scripted_call(agent_id, lines)` from `assemblyai_agents.drive` is the matching
test driver: it opens a real session with device audio off, sends each line as
a user turn, and hands back a `Transcript` with `.spoken` and `.agent_lines`.

Nothing in the SDK knows what a tunnel is. How the developer's machine becomes
reachable is their choice; `examples/e2e_check.py` will start ngrok or
cloudflared as a convenience, and takes `--public-url` when they already have
an address.

### Deterministic script, or a model?

For a regulated script the answer is both, split by decision rather than by
phase. Let a model do the understanding, which is what it is good at: what did
the caller just say, which branch is this, what values did they give. Keep the
words and the ordering in code: which line is legal next, whether a required
disclosure has been read, whether verification passed, what gets charged. A
model that recites a legally required sentence is a liability with no upside,
and a keyword state machine that has to classify free speech is a stream of
regexes that never quite closes.

The seam that makes this work is that the position a caller is entitled to hear
is settled in code, and only its delivery goes to the model: hand it the
official wording plus what the caller actually said, and have it convey the one
while answering the other. Keep the canned text as the fallback for when the
model is slow or unavailable, and give the model call a timeout well inside the
platform's ten seconds.

In practice that is three model calls and no more: classify what the caller
wants when keywords cannot; judge a yes or no that the words do not settle
("uh-huh" is a yes, and a consent question asked twice is how a call starts to
loop); and deliver a settled position. `examples/starter/model.py` is those
three, with an off switch, and every test in the starter runs with the model
off so the fallback wording stays honest.

A working endpoint, tools and replies in one service, is `byo_llm_server.py` in
the SDK repo's `examples/`. `scripts/e2e_check.py` exercises it: with the
backend behind `--forward`, the report shows the platform's
`POST /v1/chat/completions`, the tool call the endpoint asked for, and the
second completion carrying the result.

## Telephony

- Managed number: `client.phone_numbers.purchase_available(PurchaseAvailablePhoneNumberRequest(country_code="US", number_type=NumberType.local, area_code=415, agent_id=...))`. Billable: confirm with the user before running it.
- Own number: `import_(ImportPhoneNumberRequest(phone_number, termination_uri))` then `assign_agent(number, PhoneNumberAssignAgentRequest(agent_id=...), agent=agent)`.
- Outbound: `client.calls.create(CreateCallRequest(from_number, to_number))`; `from_number` must be on the account with an agent assigned.
- Human transfer: `transfer_targets=[HumanTransfer(name, phone_number, mode="cold"|"warm", ...)]` **requires** `outbound_trunk_id`. Consult fields are warm-only. E.164 everywhere (`+14155550123`).
- Telephony audio is `audio/pcmu`/`audio/pcma`; the default `audio/pcm` at 24 kHz is for WebSocket clients. Transfer targets and pre-connect do nothing on a WebSocket session.
- **Keypad (DTMF) capture blocks WebSocket sessions entirely.** A tool with
  `dtmf_collected_arguments` makes the platform refuse the session with
  `invalid_value: tool '…' collects '…' from the phone keypad (DTMF), which only
  exists on telephony calls`, so the connection closes 1008 and nothing runs.
  When the user wants both keypad capture and a terminal test, put the profiles
  behind a flag and deploy two shapes from one declaration: with them for the
  phone number, without them for the WebSocket run.
- **Every `DtmfCollectionProfile` must state `sensitive`.** The generated model
  types it optional, but the API rejects a profile that leaves it out
  (`sensitive: must be stated`). `True` suppresses every spoken and stored trace
  of the value, which is what a card number or an account number needs; `False`
  allows read-back, which is fine for an expiry date.

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
| `session.error invalid_value` naming a tool that "collects … from the phone keypad" | That agent cannot run over WebSocket at all. Deploy a variant without `dtmf_collected_arguments` to test from a terminal. |
| `ValidationError: … dtmf_collected_arguments[n].sensitive: must be stated` | Set `sensitive=True` or `False` explicitly on every keypad profile. |
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
