# assemblyai-agents

Backend SDK for the [AssemblyAI Voice Agents API](https://www.assemblyai.com/docs),
in Python. Declare an agent and its tools in code, deploy it with one call, serve
the tool and pre-connect logic from your own backend, receive webhooks, and put
the agent on a phone number.

```python
from assemblyai_agents import Client, VoiceAgent, tool
from assemblyai_agents.models.rest import HttpMethod, PlaintextHttpToolConfig

@tool(
    timeout_seconds=10,
    http=PlaintextHttpToolConfig(url="https://api.example.com/tools/lookup_order", http_method=HttpMethod.POST),
)
async def lookup_order(order_id: str) -> dict:
    """Look up the status of a customer's order by its order number.

    Args:
        order_id: The order number the caller read out, like W004.
    """
    return await orders.status(order_id)   # your code; served by your backend

agent = VoiceAgent(
    name="Pizza Line",
    voice="ivy",
    system_prompt="You answer order-status questions for Pizza Palace. Keep it short.",
    greeting="Pizza Palace, how can I help?",
    tools=[lookup_order],
)

deployed = Client().agents.create(agent)
print(deployed.id)  # agent_...
```

What is in the box:

- **`VoiceAgent` + `@tool`** – declare an agent and its tools as ordinary
  Python. Parameter schemas are derived from type hints and docstrings, and
  every rule the server would reject is checked before the request leaves.
- **REST client** (sync and async) for agents, sessions, calls, phone numbers,
  short-lived tokens, webhook subscriptions and the built-in tool catalog, with
  retries, idempotency keys and a typed exception hierarchy.
- **Backend contracts** for the HTTP tools and pre-connect requests the platform
  calls on your service, plus **webhook signature verification**.
- **Telephony helpers** (`HumanTransfer`, `PreConnectRequest`, keypad input).
- **Realtime WebSocket client** (`AgentConnection`, `AsyncRealtimeSession`) for
  talking to a deployed agent from a terminal, testing, or building your own
  audio transport.
- An offline **`testing`** module for unit-testing your tools.

## How it fits together

```
 caller ──phone / WebSocket──▶  AssemblyAI Voice Agents platform  ──HTTPS──▶  your backend
                                 speech-to-text, LLM, text-to-speech           tool endpoints
                                 turn-taking, transfers, recordings            pre-connect lookups
                                        │                                      webhook receiver
                                        └── REST API ◀── this SDK ── deploy agents, numbers, webhooks
```

You own the agent's definition and its business logic; the platform owns the
call. When the model decides to use a tool, the platform POSTs the arguments to
the URL you configured on that tool and speaks the result. Before a phone
call is answered it can hit your pre-connect endpoint to personalise the
conversation,
and when a session or call ends it delivers a signed webhook. Pre-connect,
transfers and keypad input apply to phone calls; tools and webhooks apply to
every session.

## Requirements

- Python 3.11 or newer
- An AssemblyAI API key
- Only for microphone/speaker audio from a terminal: the PortAudio system
  library (`brew install portaudio` on macOS, `apt install portaudio19-dev` on
  Debian/Ubuntu)

## Installation

The package is distributed from this Git repository rather than PyPI. `pip`
(and `uv`) install straight from the repository URL:

```bash
pip install "git+https://github.com/dan-ince-aai/assemblyai-agents-python.git"
```

Pin to a release tag or commit so your builds are reproducible:

```bash
pip install "git+https://github.com/dan-ince-aai/assemblyai-agents-python.git@v0.1.0"
```

In a `requirements.txt`:

```text
assemblyai-agents @ git+https://github.com/dan-ince-aai/assemblyai-agents-python.git@v0.1.0
```

In a `pyproject.toml`:

```toml
dependencies = [
  "assemblyai-agents @ git+https://github.com/dan-ince-aai/assemblyai-agents-python.git@v0.1.0",
]
```

With `uv`:

```bash
uv add "assemblyai-agents @ git+https://github.com/dan-ince-aai/assemblyai-agents-python.git"
```

With microphone and speaker support for the terminal client:

```bash
pip install "assemblyai-agents[audio] @ git+https://github.com/dan-ince-aai/assemblyai-agents-python.git"
```

If you were given access to a **private** copy of this repository, install over
SSH with a GitHub account that has been granted access:

```bash
pip install "git+ssh://git@github.com/dan-ince-aai/assemblyai-agents-python.git"
```

Check the install:

```bash
python -c "import assemblyai_agents; print(assemblyai_agents.__version__)"
```

## Authentication and regions

Set `ASSEMBLYAI_API_KEY` in the environment and the client picks it up, or pass
it explicitly:

```python
from assemblyai_agents import Client, AsyncClient

client = Client()                          # reads ASSEMBLYAI_API_KEY
client = Client(api_key="...")             # or pass it
aclient = AsyncClient(timeout=60, max_retries=5)
```

The default base URL is `https://agents.assemblyai.com`. A US-hosted deployment
is available at `https://agents.us.assemblyai.com`. Agents, phone numbers and
sessions are stored per host, so keep an agent and everything attached to it on
one of them:

```python
client = Client(base_url="https://agents.us.assemblyai.com")
```

## Quickstart

Runnable agents, a starter project and a Claude Code skill are in a separate
repository, because they are files you clone and edit while this is a package
you install:

**https://github.com/dan-ince-aai/assemblyai-agents-examples**

The shape they all share: an agent is a script that serves your own functions,
plus a declaration pointing the platform at it. There is no service to write.

```python
from assemblyai_agents import Client, VoiceAgent, tool
from assemblyai_agents.models.rest import HttpToolHeaderInput
from assemblyai_agents.serving import claim_port, serve

@tool(timeout_seconds=10)
async def lookup_order(order_id: str) -> dict:
    """Look up the status of a customer's order by its order number.

    Args:
        order_id: The order number the caller read out, like W004.
    """
    return await orders.status(order_id)      # your database, your API, anything

TOOLS = [lookup_order]

def build(base_url: str) -> VoiceAgent:
    auth = HttpToolHeaderInput(name="Authorization", value=f"Bearer {SECRET}")
    return VoiceAgent(
        name="Pizza Line",
        voice="ivy",
        system_prompt="You answer order-status questions for Pizza Palace.",
        greeting="Pizza Palace, how can I help?",
        tools=[t.hosted_at(f"{base_url}/tools/{t.name}", headers=[auth]) for t in TOOLS],
    )

claim_port(port=8000)                         # before the deploy, not after
agent = build(PUBLIC_BASE_URL)                # a public HTTPS address
agent_id = Client().agents.create(agent).id   # persist this and reuse it
serve(agent, port=8000, tool_secret=SECRET)   # blocks; the platform calls in
```

`http=` cannot be passed to `@tool` when the address is a tunnel, because the
decorator runs at import and the address does not exist yet. `hosted_at` returns
a new tool bound to an address and leaves the original alone, so a module-level
`TOOLS` stays importable by tests.

`serve()` answers every route the platform will call, read off the declaration,
on the standard library alone: `POST /tools/{name}` per tool, the reply
endpoint when you pass `reply=`, a route per pre-connect request, webhook
delivery and `/healthz`. `routes()` returns the same handlers as plain callables
to mount into an application you already have.

`claim_port()` fails before the deploy rather than after it. Deploying repoints
the stored agent, so a port still held by an earlier run would otherwise leave a
live agent whose tool URLs answer to nothing.

## Declaring tools

`@tool` turns a function into a `Tool`: the handler plus the wire definition
the API needs. The rules below are enforced at decoration time with a
`ConfigurationError` that says what to change, so a bad tool fails when the
module is imported rather than after a round trip.

```python
from typing import Literal, Optional
from pydantic import BaseModel
from assemblyai_agents import tool
from assemblyai_agents.models.rest import ResponseInstructions

class Address(BaseModel):
    line1: str
    city: str
    postcode: str

@tool(
    timeout_seconds=15,
    http=hosted("/tools/schedule_delivery"),
    response_instructions=ResponseInstructions(
        success="Confirm the delivery window out loud.",
        error="Apologise and offer to take a phone number for a call back.",
    ),
)
def schedule_delivery(
    order_id: str,
    address: Address,
    window: Literal["morning", "afternoon", "evening"] = "afternoon",
    notes: Optional[str] = None,
) -> dict:
    """Book a delivery slot for an order that has already been paid for.

    Args:
        order_id: The order number, like W004.
        address: Where the order should be delivered.
        window: Preferred part of the day.
        notes: Anything the driver should know.
    """
    return {"order_id": order_id, "window": window, "confirmed": True}
```

- **Name.** The function name is the tool name the model calls; it must be
  `snake_case` and must not collide with an AssemblyAI platform tool
  (`aai_credit_card_luhn_check`, `aai_pre_connect_context`).
- **Description.** The first paragraph of the docstring is what the model reads
  to decide whether to call the tool. It is required. An `Args:` section
  provides per-parameter descriptions.
- **Parameters.** Every parameter needs a type hint. Supported: `str`, `int`,
  `float`, `bool`, `list[T]`, `dict[str, T]`, `Literal[...]`, an `Enum`
  subclass, `Optional[T]` / `T | None`, and pydantic `BaseModel` subclasses (nested models
  are inlined; recursive models are refused). A parameter with a default is
  optional in the schema. `*args`, `**kwargs` and positional-only parameters
  cannot be described and are refused.
- **Return type.** A return annotation is required and must be
  JSON-serialisable: `dict`, `list`, `str`, `int`, `float`, `bool`, `None` or a
  pydantic model.
- **Sync or async.** Both work. A sync handler run through `Tool.invoke` is
  moved to a worker thread so it never blocks an event loop.
- **Options.** `timeout_seconds` (1–300, default 120: lower it for anything a
  caller waits through), `http=` (where the platform calls, see below),
  `response_instructions` (static text appended to the model's guidance after
  success/failure), `dtmf_collected_arguments` (collect a parameter from the
  phone keypad instead of speech), and `execution_mode` (only `interactive` is
  available in v1).

Calling a `Tool` calls the underlying function unchanged, so tools stay directly
testable. `tool.definition()` returns the wire model and `tool.spec` the parsed
description, schema and handler metadata.

### Where a tool runs

A tool declared **with `http=`** is served by your backend: the platform calls
that HTTPS endpoint itself, so it works for phone calls and for any client.
`GET`/`DELETE` send the arguments as query parameters; `POST`/`PUT`/`PATCH` send
them as a JSON body. The response body is handed to the model as the result.
Header values (for example an `Authorization` header your backend checks) are
stored encrypted and never returned by the API. Pass `http_method` explicitly
(`HttpMethod.POST`, `HttpMethod.GET`, …). The API verifies when an agent is
created or updated that every tool and pre-connect hostname resolves in public
DNS, so a placeholder URL is rejected; during development point the URLs at a
tunnel (see *One file, no backend* above).

A tool declared **without** `http=` is a different thing, and almost certainly
not what you want: the platform hands the call to whichever process is holding
a WebSocket session, so the tool only exists while a browser or a desktop app
is connected. A phone call has nobody to hand it to, and the SDK refuses to
attach a number to an agent that still has one.
`agent.client_resident_tool_names()` lists them.

### `ToolContext`

A tool may declare one parameter annotated `ToolContext`. It is recognised by
annotation, not by name, and is excluded from the schema. The context offers
`http` (an async HTTP client), `log`, `session_id`, `aborted` and
`secret(name)`. Supply your own object satisfying the protocol when serving the
tool (`tool.invoke(context=ctx, **arguments)`); calling `invoke` without
`context=` on a tool that declares one raises `TypeError` unless the parameter
has a default (`ctx: ToolContext = None`). The `testing` module ships an
offline double. Handlers routed through `AgentConnection(tools=...)`
receive the model's arguments only.

## Pre-connect requests

Up to two HTTPS calls made before a **phone call** is answered, typically to
look the caller up in your CRM. Values captured from one can be sent to the
next and can override the greeting. The captured values are exposed to the
model through the `aai_pre_connect_context` platform tool. Like transfer
targets and keypad input, pre-connect is a telephony feature and is inert on a
WebSocket session.

```python
from assemblyai_agents import Captured, Header, PreConnectRequest

agent = VoiceAgent(
    ...,
    pre_connect=[
        PreConnectRequest(
            url="https://api.example.com/pre-connect/whois",
            headers=[Header(name="Authorization", value="Bearer ...")],
            returns=[Captured(name="customer_tier", path="customer.tier", default="standard")],
            timeout_ms=400,
            allow_overrides=True,   # a top-level "greeting" key in the response replaces the greeting
        ),
        PreConnectRequest(url="https://api.example.com/pre-connect/tier", sends=["customer_tier"]),
    ],
)
```

Your endpoint has to answer within the request's timeout (800 ms ceiling per
request; `timeout_ms` only lowers it). Pre-connect **fails open**: a timeout or
error means the call proceeds without the values. A response with a top-level
`"reject": true` aborts the call, which is the one thing a pre-connect endpoint
can do to stop a conversation. `sends` may only name values captured by an
earlier entry; the SDK checks the order before deploying.

## Phone calls

### Give an agent a phone number

```python
from assemblyai_agents.models.rest import NumberType, PurchaseAvailablePhoneNumberRequest

number = client.phone_numbers.purchase_available(
    PurchaseAvailablePhoneNumberRequest(
        country_code="US", number_type=NumberType.local, area_code=415,
        agent_id=deployed.id,
    )
)
print(number.phone_number)   # +1415...
```

Bring your own number by pointing your carrier's SIP trunk at AssemblyAI and
importing it, then assigning an agent:

```python
from assemblyai_agents.models.rest import ImportPhoneNumberRequest, PhoneNumberAssignAgentRequest

client.phone_numbers.import_(ImportPhoneNumberRequest(
    phone_number="+14155550123",
    termination_uri="example.pstn.twilio.com",
))
client.phone_numbers.assign_agent(
    "+14155550123",
    PhoneNumberAssignAgentRequest(agent_id=deployed.id),
    agent=agent,   # optional: lets the SDK refuse a declaration with client-resident tools
)
```

Other operations: `list()`, `get(number)`, `unassign_agent(number)`,
`deregister(number)`, and `purchase(PurchasePhoneNumberRequest(phone_number=...))`
for a specific number.

### Place an outbound call

```python
from assemblyai_agents.models.rest import CreateCallRequest

call = client.calls.create(CreateCallRequest(from_number=number.phone_number, to_number="+12125550148"))
print(call.id, call.status)          # dialing
print(client.calls.get(call.id).status)
```

The `from_number` must be registered to the account with an agent assigned; that
agent handles the call. `client.calls.list(status=..., direction=...)` pages
through call history.

### Transfers to a human

```python
from assemblyai_agents import HumanTransfer

agent = VoiceAgent(
    ...,
    outbound_trunk_id="trunk_...",     # required whenever a human transfer target exists
    transfer_targets=[
        HumanTransfer(name="front desk", phone_number="+14155550100"),                 # cold
        HumanTransfer(name="on-call", phone_number="+14155550101", mode="warm",
                      consult_instructions="Summarise the caller's issue in one sentence.",
                      consult_timeout=45, record_consult=False),
    ],
)
```

Numbers must be E.164. The consult fields only apply to a warm transfer and are
refused on a cold one. Transfer targets are ignored on WebSocket sessions.

### Keypad (DTMF) input

Collect a parameter from the phone keypad rather than speech, for example a
card number:

```python
from assemblyai_agents.models.rest import DtmfCollectionProfile

@tool(
    http=hosted("/tools/take_payment"),
    timeout_seconds=120,        # keypad entry plus a confirmation takes a while
    dtmf_collected_arguments=[
        DtmfCollectionProfile(
            parameter_name="card_number", min_digits=15, max_digits=16,
            sensitive=True,     # required on every profile, see below
            confirm=True,
            prompt="Using your keypad, enter your card number, then press pound.",
        )
    ],
)
async def take_payment(card_number: str, amount: float) -> dict:
    """Charge the caller's card for the order total."""
    ...
```

Two rules the generated model does not express:

- **`sensitive` must be stated** on every profile, `True` or `False`. Leaving it
  out is rejected with `sensitive: must be stated`. `True` suppresses every
  spoken and stored trace of the value, which is what a card number needs.
- **A tool with keypad profiles cannot run over WebSocket.** The platform
  refuses the session with `invalid_value: tool '…' collects '…' from the phone
  keypad (DTMF), which only exists on telephony calls` and closes it. If you
  also want to drive the agent from a terminal, put the profiles behind a flag
  and deploy two shapes from the one declaration.

## Webhooks

Subscribe to `session.started`, `session.completed`, `call.connected`,
`call.ended` and `call.failed`:

```python
from assemblyai_agents.models.rest import CreateWebhookSubscriptionRequest, WebhookEvent

sub = client.webhooks.create(CreateWebhookSubscriptionRequest(
    url="https://api.example.com/webhooks/voice-agents",
    events=[WebhookEvent.session_completed, WebhookEvent.call_ended],
    secret="a-random-secret-of-at-least-32-characters",
    agent_id=deployed.id,    # optional: only this agent's events
))
```

Every delivery carries an `X-AAI-Signature` header. Verify it against the exact
raw request body **before** parsing any JSON:

```python
from assemblyai_agents import verify, WebhookVerificationError

@app.post("/webhooks/voice-agents")
async def webhook(request: Request):
    body = await request.body()
    try:
        event = verify(body, request.headers.get("X-AAI-Signature", ""), WEBHOOK_SECRET)
    except WebhookVerificationError as exc:
        raise HTTPException(400, str(exc))
    ...
```

`client.webhooks.list_deliveries(session_id)` and
`list_latest_deliveries(session_id)` show what was delivered for a session;
`list()`, `get()`, `update()` and `delete()` manage subscriptions.

## Audio, transcription and turn detection

`input=` and `output=` take typed helpers that emit exactly what the stored
agent carries:

```python
from assemblyai_agents import AudioFormat, AudioInput, AudioOutput

agent = VoiceAgent(
    name="Pizza Line",
    voice="ivy",
    system_prompt="...",
    input=AudioInput(
        format=AudioFormat(encoding="audio/pcm", sample_rate=24000),
        keyterms=["margherita", "calzone", "W004"],           # up to 100
        transcription_mode="balanced",                         # or min_latency / max_accuracy
        language_codes=["en"],
        voice_focus="near-field",                              # or far-field
        extra={"turn_detection": {"min_silence": 600, "max_silence": 2500}},
    ),
    output=AudioOutput(volume=90.0),
)
```

- Encodings: `audio/pcm` (16-bit little-endian mono; `sample_rate` must be
  24000 and is only valid on this encoding), `audio/pcmu` and `audio/pcma`
  (8 kHz telephony codecs).
- Turn detection is passed through `extra`. Fields: `min_silence` (ms, default
  1000), `max_silence` (ms, default 3000), `interrupt_response` (default
  `True`), `interruption_delay`, `vad_threshold` (default 0.5).
- `extra` refuses any key the class already models, so a value can never be set
  twice with one silently winning.
- The voice is set once, at the top level (`voice="ivy"`); `AudioOutput` does
  not model it.

## Bring your own LLM

Response generation can move to your backend too. Point the agent at any
OpenAI-compatible chat-completions endpoint and the platform asks *it* what to
say on every turn, while still handling speech, turn-taking and telephony. The
key is write-only and never returned.

```python
from assemblyai_agents.models.rest import LlmConfigRequest

agent = VoiceAgent(
    ...,
    llm=LlmConfigRequest(
        base_url="https://api.example.com/v1",   # or https://api.openai.com/v1
        model="pizza-line-rules",                # whatever your endpoint expects
        api_key="...",                           # sent as Authorization: Bearer
    ),
)
```

### `assemblyai_agents.replies` reads the request and answers it

Reading the transcript and streaming Server-Sent Events is contract detail, not
your agent. `replies` is that detail and nothing else: thirteen names, no
framework, no opinion about how you decide.

```python
from assemblyai_agents.replies import Turn, call_tool, say, silence, stream

def decide(turn):
    if turn.pending and turn.pending.name == "verify_caller":
        return say("Thanks, how can I help?") if turn.pending.get("verified") else say("Try again?")
    if not turn.caller_said:
        return say("Could you give me your full name?")
    return call_tool("verify_caller", caller_said=turn.caller_said)

@app.post("/v1/chat/completions")          # any framework; this one is FastAPI
async def replies(request: Request):
    body = await request.json()
    turn = Turn.from_request(body)
    return StreamingResponse(stream(turn, decide(turn)), media_type="text/event-stream")
```

Three functions say what happens next: `say(text)`, `call_tool(name, **args)`
and `silence()`, which is how a finished call ends since an agent cannot hang
up. `call_tool` drops arguments the conversation never established, so the
platform accepts the call.

`Turn` is the request already read, with the traps handled:

| | |
| --- | --- |
| `turn.caller_said` | the caller's latest words, from a user message or a quoted instruction |
| `turn.pending` | the tool result nothing has been said about yet, which is the cue to speak |
| `turn.pending.ran` | `False` when the platform refused the call, so a refusal is never reported as a result |
| `turn.preconnect` | the pre-connect captures, read out of the tool result the platform injects |
| `turn.result_of(name)` | an earlier result, to read back rather than call again |
| `turn.answer_following("your postcode?")` | a value you collected over several turns |
| `turn.said_before(line)` | with the caveat that an interrupted turn does not always come back |
| `digits_said("four four seven one")` | `"4471"`, and `"forty one eleven"` gives `"4111"` |

How you organise `decide` is up to you. The starter shows one way, as forty
lines of ordered stages in its own file, because that is an opinion and
opinions belong in an example rather than in the SDK.

### Worked endpoints

One file with tools and replies together, subagent routing with a different
model per stage, and a starter project with an offline rehearsal harness are all
in the examples repository: https://github.com/dan-ince-aai/assemblyai-agents-examples

### What the platform sends your endpoint

Captured from a live session, so build against this rather than the OpenAI docs
alone:

- `POST {base_url}/chat/completions`, `Authorization: Bearer <your api_key>`,
  `User-Agent: LiveKit Agents/...`, and a 10 second read timeout, so get the
  first chunk out fast and do slow work in a tool.
- `stream: true` on every call, with `stream_options: {"include_usage": true}`.
  Server-Sent Events are required; a plain JSON body will not do.
- `messages[0]` is your `system_prompt` **with the platform's own spoken-output
  guidance appended** (no formatting characters, how to say identifiers, dates
  and emails aloud, when to prefer a tool over asking). The greeting arrives as
  an `assistant` message.
- `tools` carries the agent's tools in OpenAI function form with
  `tool_choice: "auto"`, except the platform nests a second `type: "function"`
  plus its own `timeout_seconds` and `execution_mode` inside `function`. Read
  the name from `tool["function"]["name"]`.
- Emit `tool_calls` and the platform runs the tool, then calls you again with a
  `tool` message carrying the result and its `tool_call_id`, followed by a
  `system` note such as "The function call … has just completed". So the tool
  message is usually **not** the last one: treat "a tool result with no
  assistant text after it" as the cue to answer, and read a repeated call's
  answer back out of the transcript instead of asking for it again. A failed
  call comes back with coaching text appended, and after three consecutive
  failures the platform tells you to stop retrying.
- **Arguments must be values the call established.** The platform checks each
  one against the conversation and refuses to run the tool otherwise, returning
  a note that says so: "The call has not established a value for `account_ref` …
  Never invent a value." An empty string counts as invented, so omit an unknown
  optional argument rather than sending `""`. Values the caller spoke, or that
  an earlier tool returned, are accepted.
- **Pre-connect captures arrive here too**, as an `aai_pre_connect_context` tool
  result at the top of the transcript:
  `{"variables": {"account_ref": "…", "consumer_first_name": "…"}}`.
- **Keypad-collected parameters are hidden from you.** The platform strips them
  from the tool schema it shows your endpoint and collects them itself. A
  collection that ends early returns prose rather than the tool's JSON, so a
  short entry is not a declined card.
- A `tool` message is therefore not always JSON. Parse defensively.

## Sessions, recordings and transcripts

```python
for s in client.sessions.list(agent_id=deployed.id, status="completed"):
    print(s.id, s.duration_seconds, s.public_close_reason)

session = client.sessions.get("sess_...")
for artifact in session.artifacts or []:
    print(artifact.type, artifact.content_type, artifact.url)
```

Once a session completes it carries three artifacts as presigned URLs: `audio`
(an Ogg recording), `timeline` (JSON: every turn with what triggered it, such as
`reply_create` or `tool_result`, which is the place to look when a tool call
went wrong) and `metadata` (JSON).

```python
# a failed hosted tool never surfaces on the client; the timeline shows it
import httpx
timeline = next(a for a in session.artifacts if a.type == "timeline")
print(httpx.get(timeline.url).json())
```

## The realtime WebSocket

`AgentConnection` connects to a deployed agent from a terminal: it mints a
short-lived token, opens the WebSocket, binds the agent, streams the microphone
in and plays replies out (with barge-in), and answers `tool.call` events for
client-resident tools with the functions in `tools=`. Callbacks: `on_ready`,
`on_user_transcript`, `on_agent_transcript`, `on_agent_delta`, `on_agent_audio`,
`on_error`. Pass `audio=False` to disable device audio and handle `reply.audio`
events yourself.

For your own transport (a browser, a telephony bridge, a test harness) use the
session it is built on:

```python
import asyncio
from assemblyai_agents import AsyncClient, base64_to_pcm
from assemblyai_agents.models.ws import (
    ReplyAudio, SessionEnded, SessionReady, ToolCall, TranscriptAgent, TranscriptUser,
)

async def main():
    async with AsyncClient() as client:
        token = (await client.tokens.create()).token
        async with await client.sessions.connect(token=token, auto_resume=True) as session:
            await session.update(agent_id="agent_...")
            async for event in session:
                match event:
                    case SessionReady():
                        print("ready", event.session_id)
                    case TranscriptUser():
                        print("you:", event.text)
                    case TranscriptAgent():
                        print("agent:", event.text)
                    case ReplyAudio():
                        pcm = base64_to_pcm(event.data)   # 24 kHz 16-bit mono PCM
                    case ToolCall():                      # only for client-resident tools
                        result = await lookup_order(**event.arguments)
                        await session.send_tool_result(event.call_id, str(result))
                    case SessionEnded():
                        break

asyncio.run(main())
```

Client → server methods: `update(...)`, `send_audio(pcm_bytes)`,
`send_tool_result(call_id, result, is_error=False)`, `send_message(text, role="user")`,
`create_reply(instructions=None)`, `cancel_reply(reply_id)`, `resume(session_id)`,
`end()`. `send_message` appends a message to the conversation history and
`create_reply` asks the agent to speak, optionally steered by `instructions`;
neither is needed on a normal audio session.

Server → client events (all pydantic models in `assemblyai_agents.models.ws`):

| Event type | Model | Notes |
| --- | --- | --- |
| `session.ready` | `SessionReady` | `session_id`, `resume_token`, the effective `config` |
| `session.updated` | `SessionUpdatedEvent` | after a successful `update` |
| `session.error` | `SessionError` | `code`, `message`, `param` |
| `session.ended` | `SessionEnded` | durations |
| `input.speech.started` / `.stopped` | `InputSpeechStarted` / `InputSpeechStopped` | barge-in cue |
| `reply.started` / `reply.audio` / `reply.done` | `ReplyStarted` / `ReplyAudio` / `ReplyDone` | `ReplyAudio.data` is base64 PCM |
| `tool.call` | `ToolCall` | `call_id`, `name`, `arguments` |
| `transcript.user` | `TranscriptUser` | final user turn |
| `transcript.agent` / `.delta` | `TranscriptAgent` / `TranscriptAgentDelta` | final and streaming agent text |

Anything the SDK does not recognise arrives as an `UnknownEvent` with the raw
payload, so a newer server never breaks the loop.

- **Auto-resume.** With `auto_resume=True`, an abnormal disconnect is retried
  with backoff for up to 25 s and the session is resumed with its
  `resume_token`; terminal errors such as `session_expired` are raised as
  `RealtimeError`.
- **Inline configuration.** You do not need a stored agent: `session.update(
  system_prompt=..., greeting=..., tools=[...], input=..., output=...,
  webhook=...)` configures the session directly. `agent_id` cannot be combined
  with other fields in the same update.
- **Tokens for browsers and apps.** `client.tokens.create()` mints a short-lived
  token (60 s to connect by default, up to 600) that a front end can use on the
  WebSocket handshake instead of your API key.
- **Audio helpers.** `pcm_to_base64`, `base64_to_pcm`, `pcm16_to_ulaw`,
  `ulaw_to_pcm16`, `pcm16_to_alaw`, `alaw_to_pcm16` for telephony codecs, and
  `microphone_stream(session)` / `PlaybackSink` for device audio.

## Errors, retries and idempotency

Every API failure is an `APIError` subclass keyed on the HTTP status, so an error
code this version has never seen still lands on the right class:

| Status | Exception |
| --- | --- |
| 400, 405 | `BadRequestError` |
| 401 | `AuthenticationError` |
| 404 | `NotFoundError` |
| 409 | `ConflictError` |
| 422 | `ValidationError` |
| 5xx | `ServerError` |
| 2xx with a non-JSON body | `ResponseError` |

Each carries `status`, `code` (compare with the `ErrorCode` constants, e.g.
`ErrorCode.AGENT_NOT_FOUND`), `message`, `param`, `request_id`, `errors` and the
`raw` response. Problems caught before a request is sent raise
`ConfigurationError`; WebSocket failures raise `RealtimeError`.

```python
from assemblyai_agents import ErrorCode, NotFoundError

try:
    client.agents.get("agent_missing")
except NotFoundError as exc:
    assert exc.code == ErrorCode.AGENT_NOT_FOUND
    print(exc.request_id)
```

- **Retries.** 408, 429, 5xx and a 409 `idempotency_in_progress` are retried up
  to `max_retries` (default 3) with exponential backoff and jitter, honouring
  `Retry-After` up to 60 s. Connection errors are retried the same way.
- **Idempotency.** Calls that create billable side effects
  (`calls.create`, phone number purchase and import) mint an `Idempotency-Key`
  once per logical request and reuse it across retries. Pass your own via
  `headers={"Idempotency-Key": ...}` on `client.request` to control it.
- **Escape hatch.** `client.request(method, path, json=..., params=...)`
  returns the decoded JSON of any endpoint with the same auth, retries and error
  handling; `client.request_raw(...)` returns the undecoded `RawResponse`.
- **Pagination.** Every `list()` returns a pager: iterate it directly for all
  items, or call `next_page()` for one page at a time.

## Testing your tools

`assemblyai_agents.testing` runs a tool with no network at all:

```python
import pytest
from assemblyai_agents import ToolContext, VoiceAgent, tool
from assemblyai_agents.testing import create_tool_context, get_tool

@tool
async def lookup_order(order_id: str, ctx: ToolContext) -> dict:
    """Look up one of the caller's orders by its ID."""
    response = await ctx.http.get(
        f"https://api.pizzapalace.com/orders/{order_id}",
        headers={"authorization": ctx.secret("orders_api_key")},
    )
    return response.json()

agent = VoiceAgent(name="Pizza Line", voice="ivy", system_prompt="...", tools=[lookup_order])

@pytest.mark.asyncio
async def test_lookup_order():
    ctx = create_tool_context(secrets={"orders_api_key": "test-key"})
    ctx.http.stub("GET", "https://api.pizzapalace.com/orders/W004", json={"status": "shipped"})

    result = await get_tool(agent, "lookup_order").invoke(context=ctx, order_id="W004")

    assert result == {"status": "shipped"}
    assert ctx.http.calls[0].headers["authorization"] == "test-key"
```

An unstubbed request is refused with an error that lists what *is* stubbed, so a
tool cannot reach the network by accident. Assert on `agent.to_request()` to
pin the exact payload a declaration produces.

## Sync and async

`Client` and `AsyncClient` expose the same resources (`agents`, `sessions`,
`calls`, `phone_numbers`, `tokens`, `webhooks`, `builtin_tools`) with the same
method names. The realtime WebSocket (`sessions.connect`, `AgentConnection`) is
async only. Both clients are context managers and release their connection
pools on exit.

## Using this SDK with a coding agent

The Claude Code skill and an `AGENTS.md` for other coding agents live with the
examples, because they refer to those files by path:

```bash
git clone https://github.com/dan-ince-aai/assemblyai-agents-examples.git
cp -r assemblyai-agents-examples/.claude/skills/assemblyai-agents-sdk ~/.claude/skills/
```

## Development

```bash
git clone https://github.com/dan-ince-aai/assemblyai-agents-python.git
cd assemblyai-agents-python
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,audio]"
pytest
```

The test suite runs entirely offline against mocked transports. Only
`tests/test_audio_io_surface.py` imports the real `pyaudio`; it skips itself
where PortAudio is not installed.

`assemblyai_agents/models/rest.py` and `assemblyai_agents/models/ws.py` are
generated from the API's OpenAPI and WebSocket schemas. Do not edit them by
hand; they are replaced wholesale when the API changes.

## License

MIT. See [LICENSE](LICENSE).
