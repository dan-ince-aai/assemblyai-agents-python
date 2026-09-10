# `assemblyai-agents` API reference

The names in *Clients*, *VoiceAgent*, *@tool* (`tool`, `Tool`, `ToolContext`),
*Audio config*, *Telephony helpers*, *Realtime* (`AgentConnection`,
`AsyncRealtimeSession`, `ToolRouter`, `UnknownEvent`), *Webhook verification*,
*Exceptions* and *Audio helpers* are top-level exports of `assemblyai_agents`.
Every request/response model and enum — `PlaintextHttpToolConfig`,
`HttpToolHeaderInput`, `HttpMethod`, `ResponseInstructions`,
`DtmfCollectionProfile`, `ExecutionMode`, `*Request`, `*Response`, `NumberType`,
`WebhookEvent`, … — lives in `assemblyai_agents.models.rest` (only
`AgentCreateRequest`, `AgentUpdateRequest`, `LlmConfigRequest`,
`PlaintextToolDefinition` and `VoiceConfig` are also re-exported at top level).
WebSocket event models live in `assemblyai_agents.models.ws`.

Contents: [Install](#install) · [Clients](#clients) · [VoiceAgent](#voiceagent) ·
[@tool](#tool) · [Audio config](#audio-config) · [Telephony helpers](#telephony-helpers) ·
[Resources](#resources) · [Realtime](#realtime) · [Backend contracts](#backend-contracts) ·
[Webhook verification](#webhook-verification) · [Exceptions](#exceptions) ·
[Testing module](#testing-module) · [Audio helpers](#audio-helpers) · [Minimal backend](#minimal-backend)

## Install

```bash
pip install "git+https://github.com/dan-ince-aai/assemblyai-agents-python.git"           # core
pip install "assemblyai-agents[audio] @ git+https://github.com/dan-ince-aai/assemblyai-agents-python.git"  # + mic/speakers (needs PortAudio)
```
Python ≥ 3.11. Dependencies: httpx, pydantic v2, websockets.

## Clients

```python
Client(api_key=None, *, base_url="https://agents.assemblyai.com", timeout=30.0, max_retries=3, transport=None)
AsyncClient(...same...)
```
- `api_key` falls back to `ASSEMBLYAI_API_KEY`; missing → `ConfigurationError`.
- US host: `base_url="https://agents.us.assemblyai.com"`. Resources are per host.
- Resources: `.agents .sessions .calls .phone_numbers .tokens .webhooks .builtin_tools`
  (async client: same names, `await` the calls; `list()` returns an async pager).
- Escape hatch: `client.request(method, path, *, params=None, json=None, headers=None, idempotent=False, timeout=None)`
  → decoded JSON; `client.request_raw(...)` → `RawResponse(status_code, headers, content, request_id)`.
- Retries: 408, 429, 5xx, 409 `idempotency_in_progress`, transport errors; exponential backoff + jitter; `Retry-After` honoured (≤ 60 s).
- Pagers (`SyncPager`/`AsyncPager`): iterate for all items, or `next_page()` → list | None, `has_more`.
- Context managers: `with Client() as c:` / `async with AsyncClient() as c:`.

## VoiceAgent

```python
VoiceAgent(*, name: str, system_prompt: str, voice: str,
           greeting: str | None = None,
           llm: LlmConfigRequest | None = None,
           input: AudioInput | None = None, output: AudioOutput | None = None,
           tools: list[Tool] | None = None,
           transfer_targets: list[HumanTransfer] | None = None,
           pre_connect: list[PreConnectRequest] | None = None,
           outbound_trunk_id: str | None = None, caller_id: str | None = None)
```
- Frozen, keyword-only dataclass; cannot be subclassed. `system_prompt` is dedented + stripped.
- Checks at construction: unique tool names; pre-connect ordering/limits; transfers need `outbound_trunk_id`; `caller_id` E.164.
- `to_request() -> AgentCreateRequest`, `to_update_request() -> AgentUpdateRequest` (whole declaration; PUT replaces).
- `tool_definitions()`, `pre_connect_requests()`, `wire_transfer_targets()`, `client_resident_tool_names() -> tuple[str, ...]` (tools with no `http=`).
- `voice`: an AssemblyAI voice name, e.g. `"ivy"` (verified), `"james"`, `"mia"`. A 422 `ValidationError` names an unknown voice. On the wire it is `VoiceConfig(voice_id=voice)` (`AgentCreateRequest.voice.voice_id`, `AgentResponse.voice.voice_id`), so pin payloads with `VoiceConfig(voice_id="ivy")`, not `"ivy"`.
- Every tool/pre-connect URL host must resolve in public DNS at create/update time (`ValidationError` "URL host … does not resolve"); `agents.create` does accept a declaration whose tools are all client-resident.
- `llm`: `LlmConfigRequest(base_url, model, api_key)` – any OpenAI-compatible chat-completions endpoint; sent as a one-element list (capped at one in v1). `base_url` must be HTTPS and resolve in DNS; `api_key` is write-only (`LlmConfigResponse` returns `base_url` and `model` only). The platform calls `POST {base_url}/chat/completions` with `Authorization: Bearer <api_key>`, always `stream: true` + `stream_options: {"include_usage": true}` (Server-Sent Events required), a 10s read timeout, `tool_choice: "auto"`, and `tools` in OpenAI function form with a second `type: "function"` and the platform's `timeout_seconds`/`execution_mode` nested inside `function`. `messages[0]` is the agent's `system_prompt` plus the platform's spoken-output guidance; the greeting is an `assistant` message. Returned `tool_calls` are executed by the platform, which calls back with a `tool` message (`tool_call_id`) and then a `system` note — so the cue to speak is a tool result with no assistant text after it, and a repeat of the same call should be answered from the transcript. See `examples/byo_llm_server.py` in the SDK repo.

## @tool

```python
@tool                                   # or
@tool(timeout_seconds=120, execution_mode=None, response_instructions=None,
      http=None, dtmf_collected_arguments=None)
def name(param: T, ..., ctx: ToolContext) -> R: """Description.\n\nArgs:\n    param: text"""
```
Rules (violations raise `ConfigurationError` at decoration):
- name `snake_case`, not `aai_credit_card_luhn_check` / `aai_pre_connect_context`.
- docstring first paragraph required (= description); `Args:`/`Arguments:`/`Parameters:` section gives per-param descriptions (`name (type): text` – the type part is ignored).
- every parameter type-hinted; supported `str int float bool list[T] dict[str, T] Literal[...] Enum Optional[T] / T | None BaseModel`; nested models inlined; recursive or >32-deep models refused; no `*args/**kwargs/positional-only`.
- return annotation required: `dict list str int float bool None BaseModel` (or `Optional`/`Union` of those).
- `timeout_seconds` int 1–300. `execution_mode`: only `ExecutionMode.interactive` (hold refused in v1).
- `http`: `PlaintextHttpToolConfig(url: str, http_method: HttpMethod | None, headers: list[HttpToolHeaderInput] | None)`; `HttpMethod.GET|POST|PUT|PATCH|DELETE` is a plain `Enum` (not `str`) — pass it explicitly, because an omitted `http_method` stays the bare string `"POST"`, which serialises with a pydantic warning and compares unequal to `HttpMethod.POST`. `HttpToolHeaderInput(name, value=None, remove=False)`; read-back headers come as `{name, last_set_at}` (values never returned). A tool with no parameters is fine (`properties: {}`; the platform POSTs `{}`).
- `response_instructions`: `ResponseInstructions(success: str ≤500, error: str ≤500)`.
- `dtmf_collected_arguments`: `[DtmfCollectionProfile(parameter_name, min_digits, max_digits, prompt, sensitive, terminator="#", confirm=None, timeout_seconds=None, escalate_hotkey_disabled=None)]` – phone only, and two rules the model does not capture: `sensitive` **must** be stated on every profile or the API rejects it (`sensitive: must be stated`), and any tool carrying profiles makes the platform **refuse a WebSocket session** (`invalid_value … collects '…' from the phone keypad (DTMF), which only exists on telephony calls`, then close 1008). Put the profiles behind a flag so one declaration deploys with them for a phone number and without them for a terminal test. Each `parameter_name` must be a property of the tool's `parameters`, and the tool needs a timeout with room for entry plus confirmation.

`Tool` object: `tool.name`, `tool.spec` (`ToolSpec`: name, description, parameters (JSON schema), timeout_seconds, execution_mode, response_instructions, http, dtmf_collected_arguments, context_parameter, is_async, target), `tool.definition() -> PlaintextToolDefinition`, `await tool.invoke(context=None, **arguments)` (sync targets run in a thread; injects `context` into the `ToolContext` parameter — **required** when the function declares one, otherwise `TypeError: missing … 'ctx'`), `tool(*args, **kwargs)` calls the raw function (returns a coroutine for an async function, a value for a sync one).

`ToolContext` (Protocol; declare it as `ctx: ToolContext` — or `ctx: ToolContext = None` so `invoke(**args)` works without a context; `Optional[ToolContext]` is refused): `http` (async client with `.get/.post/.request(...)` returning objects with `.status_code .headers .text .json()`), `log` (`.debug/.info/.warning/.error(event, **fields)`), `session_id: str`, `aborted: bool`, `secret(name) -> str`. Constants: `RESPONSE_LIMIT_BYTES = 1 MiB`, `HTTP_TIMEOUT_SECONDS = 15`.

Platform tools (server-owned, listed by `client.builtin_tools.list()`): `aai_credit_card_luhn_check`, `aai_pre_connect_context`.

## Audio config

```python
AudioFormat(*, encoding: "audio/pcm" | "audio/pcmu" | "audio/pcma" = "audio/pcm", sample_rate: 24000 | None = None)  # sample_rate only with audio/pcm
AudioInput(*, format=None, keyterms: list[str] | None (≤100), transcription_mode: "balanced"|"min_latency"|"max_accuracy" | None,
           continuous_partials: bool | None, transcription_prompt: str | None (≤1750), language_codes: list[str] | None,
           voice_focus: "near-field"|"far-field" | None, voice_focus_threshold: float 0–1 | None, extra: dict | None)
AudioOutput(*, format=None, volume: float 0–100 | None, extra: dict | None)
```
- Both always emit `"type": "audio"`. `extra` merges unmodelled keys and refuses keys the class models.
- Turn detection: `AudioInput(extra={"turn_detection": {"min_silence": 1000, "max_silence": 3000, "interrupt_response": True, "interruption_delay": None, "vad_threshold": 0.5}})`.
- Voice is the top-level `VoiceAgent.voice`, never in `AudioOutput`.

## Telephony helpers

```python
HumanTransfer(*, name, phone_number (E.164), mode: "cold"|"warm" = "cold", ring_timeout: int 1–600 | None,
              consult_instructions: str | None, consult_timeout: int | None, record_consult: bool | None)  # consult_* warm-only
PreConnectRequest(*, url (https), method: "GET"|"POST"|"PUT"|"PATCH"|"DELETE" = "POST", headers: list[Header] | None,
                  sends: list[str] | None, returns: list[Captured] | None, timeout_ms: int 1–800 | None, allow_overrides: bool = False)
Header(*, name, value)          # value required (no name-only form)
Captured(*, name, path, default: str | None = None)   # dotted path into the JSON response, e.g. "customer.tier", "results.0.id"
```
- ≤ 2 pre-connect entries; `sends` only names captured by an *earlier* entry; capture names unique.
- `allow_overrides=True` ⇒ wire `allow_overrides=["greeting"]`.
- `agent.transfer_targets` need `agent.outbound_trunk_id`. `kind="agent"` transfers are not offered in v1.

## Resources

All `body` arguments are pydantic models from `assemblyai_agents.models.rest`.

| Resource | Methods |
| --- | --- |
| `agents` | `create(VoiceAgent \| AgentCreateRequest) -> AgentResponse`; `get(id)`; `list(limit=, cursor=) -> pager[AgentListItem]`; `update(id, VoiceAgent \| AgentUpdateRequest)`; `delete(id)` |
| `sessions` | `list(limit=, cursor=, status=, agent_id=) -> pager[SessionListItem]`; `get(id) -> SessionResponse` (`status`, `public_close_reason` e.g. `client_end`, `duration_seconds`, `config`, `artifacts[]` of `SessionArtifact(type, url, content_type)` — `audio` (audio/ogg), `timeline` (JSON: the turns, each with a `trigger` such as `reply_create` or `tool_result`) and `metadata` (JSON), as presigned URLs once the session completes); `delete(id)`; **async only** `connect(token=None, url=None, open_timeout=15.0, auto_resume=False, max_resume_attempts=5) -> AsyncRealtimeSession` |
| `calls` | `list(limit=, cursor=, status: CallStatus=, direction: CallDirection=)`; `create(CreateCallRequest(from_number, to_number)) -> CallResponse(id, status)` (idempotent); `get(id) -> CallGetResponse`; `delete(id)` |
| `phone_numbers` | `list(limit=, cursor=) -> pager[PhoneNumberResponse]`; `purchase_available(PurchaseAvailablePhoneNumberRequest(country_code, number_type: NumberType.local\|mobile\|national, area_code=, locality=, label=, agent_id=)) -> PhoneNumberResponse`; `purchase(PurchasePhoneNumberRequest(phone_number))`; `import_(ImportPhoneNumberRequest(phone_number, termination_uri=))`; `get(number)`; `deregister(number)`; `assign_agent(number, PhoneNumberAssignAgentRequest(agent_id), *, agent: VoiceAgent | None)`; `unassign_agent(number)` |
| `tokens` | `create(TokenCreateRequest(expires_in_seconds=60 (1–600)) \| None) -> TokenResponse(token, expires_at)` |
| `webhooks` | `create(CreateWebhookSubscriptionRequest(url, events: list[WebhookEvent], secret (≥32 chars), agent_id=, enabled=True))`; `list(limit=, cursor=, include_disabled=)`; `get(id)`; `update(id, UpdateWebhookSubscriptionRequest)`; `delete(id)`; `list_deliveries(session_id, limit=, cursor=)`; `list_latest_deliveries(session_id)` |
| `builtin_tools` | `list() -> BuiltinToolListResponse(builtin_tools[])` |

`WebhookEvent`: `session_started session_completed call_connected call_ended call_failed`.
`CallStatus`: `dialing active ended failed refused`. `CallDirection`: `inbound outbound`.
`AgentResponse` fields: `id name system_prompt greeting tools[ToolResponse] pre_connect_requests voice input output transfer_targets outbound_trunk_id caller_id llm created_at updated_at` (secrets/header values are never returned).

## Realtime

### `AgentConnection`

```python
AgentConnection(*, agent_id: str, api_key=None, client: AsyncClient | None = None,
                tools: Mapping[str, Callable] | None = None, audio: bool = True,
                auto_resume: bool = True, url=None, token=None)
```
- `async with conn:` mints a token, connects, `session.update(agent_id=...)`.
- `await conn.run()` – the single read loop; returns on `session.ended`. With `audio=True` starts mic capture + speaker playback (needs `[audio]`); with `audio=False` you get `reply.audio` via `on_agent_audio`.
- Callbacks (decorator or direct call, sync or async): `on_ready(SessionReady)`, `on_user_transcript(str)`, `on_agent_transcript(str)`, `on_agent_delta(str)`, `on_agent_audio(ReplyAudio)`, `on_error(SessionError)`.
- `conn.tool(name)` decorator registers a client-resident handler; `tools={name: fn}` does the same. Handlers get `**arguments`; return value is `str()`-ed back; exceptions are sent as tool errors.
- `await conn.say(text)` sends a `conversation.message` (role user). The platform's own model did not reliably act on its content, but a **BYO LLM endpoint reads the raw transcript and does see it**, so `say()` then `create_reply()` is the way to drive a multi-turn test of one, and those turns persist. Note that a `reply.create` sent while the greeting is playing replaces the greeting. To stand in for a caller in a test: wait for the first `on_agent_transcript`, then `await conn.session.create_reply(instructions='The caller just said: "…". Respond to the caller, calling your tools as needed.')`. `conn.session` exposes the underlying `AsyncRealtimeSession`.
- There is no reply-done or raw-event callback: `AgentConnection` owns the single read loop, so the six callbacks above are the only hooks. Drive `AsyncRealtimeSession` directly when you need every event.
- A bad `agent_id` does not raise from `async with conn:`; `on_error` receives `SessionError(code=agent_not_found)` and the next send raises `websockets.exceptions.ConnectionClosedError` (1008).
- Mic failure ends the session and raises `DeviceAudioError` with guidance.

### `AsyncRealtimeSession`

Obtain via `await aclient.sessions.connect(token=...)`. Methods:
`update(*, agent_id=, system_prompt=, greeting=, input=, output=, tools=, webhook=)` (`agent_id` cannot be combined with other fields; `output.type` is immutable after the first update),
`send_audio(pcm_bytes)` (base64-encodes), `send_tool_result(call_id, result: str, *, is_error=False)`,
`send_message(content, *, role="user")` (content was not reliably visible to the model in testing), `create_reply(instructions=None)` (the reliable way to inject or steer a turn), `cancel_reply(reply_id)`,
`resume(session_id)`, `end()`, `close()`; `async for event in session`; `close_code`.

Events (`assemblyai_agents.models.ws`): `SessionReady(session_id, resume_token, expires_at, config)`,
`SessionUpdatedEvent(config)`, `SessionError(code: Code, message, param)`, `SessionEnded(session_duration_seconds, audio_duration_seconds)`,
`InputSpeechStarted`, `InputSpeechStopped`, `ReplyStarted(reply_id)`, `ReplyAudio(reply_id, data: base64)`,
`ReplyDone(reply_id, status: completed|interrupted|cancelled|failed)`, `ToolCall(call_id, name, arguments)`,
`TranscriptUser(item_id, text)`, `TranscriptAgent(item_id, reply_id, text, interrupted)`, `TranscriptAgentDelta(delta)`,
plus `UnknownEvent(type, raw)` for anything unrecognised (a top-level export defined in `assemblyai_agents.realtime`, not in `models.ws`).
`Code` values include `agent_not_found unauthorized session_not_found session_expired session_forbidden invalid_config invalid_value immutable_field at_capacity concurrency_exceeded audio_rate_violation invalid_audio internal_error`.

Wire audio: 16-bit little-endian mono PCM, 24 kHz (`audio/pcm`); telephony encodings 8 kHz μ-law / A-law.

## byo: writing the replies

`assemblyai_agents.byo` — everything in *Backend contracts* below, already handled.

```python
from assemblyai_agents.byo import (
    Turn, Responder, Reply, Memo, Say, Call, Silence, ToolResult,
    sse, completion, established, digits_said, tool_runner, mount_fastapi,
)
```

- `Turn.from_request(body) -> Turn`: `.request .messages .tool_names .caller_said .spoken .preconnect .pending .results`; methods `.said_before(marker)`, `.answer_following(fragment)`, `.result_of(name, arguments=None)`, `.has(tool)`, and the verbs `.say(text)`, `.call(name, **args)`, `.silence()`.
  - `.caller_said` prefers a `user` message, falling back to text a `system` message quotes (how a test driver injects a turn); the platform's own notes use single quotes and never match.
  - `.pending` is the newest `ToolResult` with no assistant text after it — the cue to speak — and `None` once something has been said or when the result is the pre-connect context.
  - `.preconnect` is the `{"variables": {...}}` from the `aai_pre_connect_context` result the platform injects on a phone call.
- `ToolResult`: `.name .arguments .value .ran .note`, `.get(key)`, `.keypad_incomplete`. `ran=False` means the platform refused or failed the call and `note` is its prose; never report that as the tool's outcome.
- `Say(text)` / `Call(name, **arguments)` / `Silence()`. `Call` drops arguments whose value is `None` or `""`, because the platform refuses values the call never established. `established(**kwargs)` does that on its own.
- `Responder`: `.stage(name, until=None)` decorator, `.add(name, handler, until=None)`, `.decide(turn) -> action`, `.current(turn) -> Stage | None`, `.respond(body) -> Reply`, `.names`. The first stage whose `until` is unsatisfied handles the turn; a stage returning `None` falls through. `Stages` is the same class under its older name.
- `Reply`: `.stream()` (SSE lines), `.json()`, `.media_type`, `.spoken`, `.tool`, `.action`, `.stage`, and a one-line `str()` for logs.
- `sse(turn, action, chunk_words=True)` / `completion(turn, action)` if you would rather not use `Responder`.
- `Memo`: `.note(key, *markers)`, `.has(key, marker)`, `.forget(key=None)`. Process-local. Needed because an interrupted turn does not reliably reappear in the transcript, so "have I said this?" cannot be answered from the messages alone.
- `digits_said(text)` — digits from figures or words, merging "forty one" into 41 and expanding "double one".
- `tool_runner(tools)` → `async run(name, arguments)`, raising `LookupError` for an unknown tool and `ValueError` for arguments that do not fit.
- `mount_fastapi(app, responder, *, tools, tool_secret, llm_key, pre_connect=None, webhook_secret=None, prefix="", log=print)` adds `POST /v1/chat/completions`, `POST /tools/{name}`, a route per pre-connect handler, `POST /webhooks/voice-agents` and `GET /healthz`, with the auth checks. FastAPI is imported inside the function, so it is not a dependency of this package.

`assemblyai_agents.drive` — `await scripted_call(agent_id, lines, *, api_key=None, client=None, turn_timeout=45, linger=2, on_turn=None, url=None) -> Transcript`, and `call(...)` for synchronous code. `Transcript`: `.session_id .turns .errors .timed_out .agent_lines .caller_lines .spoken .ok`. Device audio is off and each line is sent as a real `user` turn, which persists and which a BYO reply engine can read.

Nothing in this package knows what a tunnel is. `examples/e2e_check.py` starts ngrok or cloudflared as a convenience and takes `--public-url` when you already have an address.

## Backend contracts

- **Unestablished values are refused.** Before running a tool the platform checks that each argument's value appeared in the call, and refuses the call otherwise with a `tool` message plus a system note ("The call has not established a value for X … Never invent a value"). `""` counts as invented: omit unknown optionals. Values from an earlier tool result, or spoken by the caller, are accepted.
- **Pre-connect context** arrives as an `aai_pre_connect_context` tool result at the top of the transcript (`{"variables": {...}}`), which is how a BYO LLM endpoint sees the captures.
- **Keypad parameters are stripped** from the tool schema shown to a BYO LLM endpoint, and that tool's `execution_mode` is set to `hold` by the platform. A collection that ends early returns prose, not the tool's JSON: "did not finish entering 'card_number' on their keypad (too_short), so the tool was not called".
- **HTTP tool** – platform → your URL with the tool's configured headers (observed `User-Agent: Python/3.13 aiohttp/…`). `POST`/`PUT`/`PATCH`: JSON body = the arguments object (`{"order_id": "W004"}`, `{}` for a no-parameter tool); `GET`/`DELETE`: arguments as query parameters, all strings — `Tool.invoke` does not coerce them, so restore numbers/booleans from `tool.spec.parameters` before calling. Respond with JSON (any shape) within `timeout_seconds`; it is stringified for the model. Non-2xx / timeout ⇒ the model is told the tool failed (and `response_instructions.error` applies); the client sees no error, only the apology. Hostnames are DNS-checked when the agent is created/updated.
- **Pre-connect** (phone calls only) – platform → `PreConnectRequest.url` with `method` and `headers`, before the call is answered; must answer within `timeout_ms` (≤ 800 ms). Body carries values named in `sends`. Response JSON: values at `Captured.path`; top-level `greeting` (needs `allow_overrides=True`) overrides the greeting; top-level `reject: true` aborts the call. Any failure ⇒ proceed without values.
- **Webhook** – platform → subscription `url`, `POST`, header `X-AAI-Signature: t=<unix>,v1=<hex hmac-sha256(secret, f"{t}." + raw_body)>`. Verify with `verify()` below; respond 2xx.

## Webhook verification

```python
verify(payload: bytes, signature_header: str, secret: str, *, tolerance: int = 300, now: int | None = None) -> dict
```
Raises `WebhookSignatureError` (bad/missing header, MAC mismatch, non-JSON body) or `WebhookTimestampError` (outside tolerance); both subclass `WebhookVerificationError`. Pass the **raw** body bytes.

## Exceptions

`AssemblyAIAgentsError` ← `ConfigurationError`, `RealtimeError(close_code)`, `WebhookVerificationError`, `APIError(status, code, message, param, request_id, errors, raw)` ← `BadRequestError` (400/405), `AuthenticationError` (401), `NotFoundError` (404), `ConflictError` (409), `ValidationError` (422), `ServerError` (5xx), `ResponseError` (2xx non-JSON).
`ErrorCode` constants (plain strings): `MISSING_AUTHORIZATION UNAUTHORIZED VALIDATION_ERROR INVALID_REQUEST AGENT_NOT_FOUND PHONE_NUMBER_NOT_FOUND CALL_NOT_FOUND SESSION_NOT_FOUND WEBHOOK_SUBSCRIPTION_NOT_FOUND PHONE_NUMBER_CONFLICT PHONE_NUMBER_NOT_AVAILABLE PHONE_NUMBER_HAS_NO_AGENT TELEPHONY_PROVIDER_ERROR IDEMPOTENCY_KEY_INVALID IDEMPOTENCY_KEY_REUSE IDEMPOTENCY_IN_PROGRESS NOT_FOUND METHOD_NOT_ALLOWED INTERNAL_ERROR` …
`DeviceAudioNotInstalledError` (ImportError) when `[audio]` is missing; `DeviceAudioError` for device failures.

## Testing module

```python
from assemblyai_agents.testing import create_tool_context, get_tool
ctx = create_tool_context(secrets={"orders_api_key": "test"}, aborted=False, session_id=None)
ctx.http.stub("GET", "https://api.example.com/orders/W004", status_code=200, json={...}, text=None, headers=None)
ctx.http.calls            # [RecordedCall(method, url, headers, json, params)] – unstubbed calls raise ConfigurationError listing what is stubbed
ctx.log.records           # [LogRecord(level, event, fields)]
ctx.secret("orders_api_key")
tool = get_tool(agent, "lookup_order")      # the Tool from agent.tools, by wire name
await tool.invoke(context=ctx, order_id="W004")   # works for sync and async tools; tool(...) is the raw function
```
Pin payloads with `assert agent.to_request() == AgentCreateRequest(...)` or `agent.to_request().model_dump(mode="json", exclude_none=True)`.

## Audio helpers

`pcm_to_base64(bytes) -> str`, `base64_to_pcm(str) -> bytes`, `pcm16_to_ulaw`, `ulaw_to_pcm16`, `pcm16_to_alaw`, `alaw_to_pcm16`;
`await microphone_stream(session, *, sample_rate=24000, frames_per_buffer=480, device=None)`; `PlaybackSink(*, sample_rate=24000, frames_per_buffer=480, device=None)` (async context manager; `.handle(event)` routes `reply.*`/`input.speech.started` for playback + barge-in).

## Minimal backend

```python
# server.py  (pip install fastapi uvicorn)
import os
from fastapi import FastAPI, HTTPException, Request, Response
from assemblyai_agents import WebhookVerificationError, verify
from agent import agent

TOOL_SECRET = os.environ["TOOL_SECRET"]; WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET")
TOOLS = {t.name: t for t in agent.tools or []}
app = FastAPI()

@app.post("/tools/{name}")
async def run_tool(name: str, request: Request):
    if request.headers.get("Authorization") != f"Bearer {TOOL_SECRET}":
        raise HTTPException(401)
    if name not in TOOLS:
        raise HTTPException(404)
    try:
        # Tools here take model arguments only. If one declares a ToolContext
        # parameter, pass your own object as invoke(context=..., **args).
        return await TOOLS[name].invoke(**await request.json())
    except TypeError as exc:            # unexpected or missing argument
        raise HTTPException(422, str(exc))

@app.post("/pre-connect/whois")
async def whois(request: Request):
    return {"customer_tier": "gold"}            # + "greeting": "..." if allow_overrides=True

@app.post("/webhooks/voice-agents")
async def webhook(request: Request):
    try:
        event = verify(await request.body(), request.headers.get("X-AAI-Signature", ""), WEBHOOK_SECRET)
    except WebhookVerificationError as exc:
        raise HTTPException(400, str(exc))
    ...  # act on event["event"]
    return Response(status_code=204)
```
Run with `uvicorn server:app --port 8000`, expose over HTTPS, set `PUBLIC_BASE_URL` to that address, deploy.
