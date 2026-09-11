# assemblyai-agents

Backend SDK for the [AssemblyAI Voice Agents API](https://www.assemblyai.com/docs),
in Python. Declare an agent and its tools as ordinary functions, decide what it
says if you want to, and deploy and serve it with one call. The platform owns
the call — speech to text, text to speech, turn-taking, telephony — and reaches
your code over HTTPS.

```python
from assemblyai_agents import VoiceAgent, tool
from assemblyai_agents.replies import Turn, call_tool, say

@tool(timeout_seconds=10)
async def lookup_order(order_said: str) -> dict:
    """Look up an order by the number the caller read out.

    Args:
        order_said: The order number exactly as the caller said it.
    """
    return await orders.status(order_said)            # your database, your API, anything

def decide(turn: Turn):
    if turn.pending and turn.pending.name == "lookup_order":
        return say(f"That order is {turn.pending.get('status')}. Anything else?")
    if not turn.caller_said:
        return say("Could you read me your order number?")
    return call_tool("lookup_order", order_said=turn.caller_said)

agent = VoiceAgent(
    name="Pizza Line",
    voice="alba",
    system_prompt="You answer order-status questions for Pizza Palace.",
    greeting="Pizza Palace, how can I help?",
    tools=[lookup_order],          # served by this process at /tools/lookup_order
    reply=decide,                  # this process decides every reply; omit it and the platform's model talks
)

agent.serve()                      # PUBLIC_BASE_URL → deploy → serve
```

Point a phone number at the agent id it prints and a real caller reaches the
function above.

## The two decisions

An agent has exactly two modes, and one field picks between them:

| | What talks | You write |
| --- | --- | --- |
| **Tools only** | the platform's model, shaped by `system_prompt` | `@tool` functions |
| **Your own replies** | your code, on every turn | `@tool` functions and a `reply=` function |

Everything else — where the tools are served, what the reply endpoint is called,
which secret the platform presents — is derived from one address, so you never
type a URL into the declaration.

## What is in the box

- **`VoiceAgent` + `@tool`** – declare an agent and its tools. Schemas come from
  type hints and docstrings, and every rule the server would reject is checked
  before the request leaves.
- **`agent.serve()` / `agent.deploy()`** – bind the declaration to your address,
  create or update the stored agent, and answer the platform's requests on the
  standard library alone. **`serving.asgi(agent)`** is the same thing as an ASGI
  app for hosts that want one (Modal, uvicorn, Lambda through Mangum).
- **`replies`** – read the platform's reply request and answer it: `Turn`,
  `say()`, `call_tool()`, `stream()`. The wire contract is handled; you write
  the decision.
- **REST client** (sync and async) for agents, sessions, calls, phone numbers,
  tokens, webhook subscriptions and the built-in tool catalog, with retries,
  idempotency keys and a typed exception hierarchy.
- **Telephony**: `PreConnectRequest` (look the caller up before answering, with
  a `handler=` this process serves), `HumanTransfer`, keypad input, outbound
  calls.
- **Webhook verification**, an **`AgentConnection`** test client for talking to
  an agent from a terminal, and an offline **`testing`** module.

## How it fits together

```
 caller ──phone / WebSocket──▶  AssemblyAI platform  ──HTTPS──▶  your process
                                speech to text, model,           /tools/{name}
                                text to speech, turn-taking      /v1/chat/completions   (reply=)
                                       │                         /pre-connect/{name}
                                       └── REST API ◀── agent.deploy()
```

The platform never sees your code. It sees a declaration — a prompt, a voice, a
list of tools with URLs, maybe a reply endpoint — and calls those URLs while a
conversation runs. `serve()` answers them from the functions on the declaration.

## Requirements

- Python 3.11 or newer
- An AssemblyAI API key in `ASSEMBLYAI_API_KEY`
- A public HTTPS address that reaches your process (see [Hosting](#hosting))
- Only for talking to an agent from a terminal: the PortAudio system library
  (`brew install portaudio` / `apt install portaudio19-dev`) and the `[audio]`
  extra

## Installation

Distributed from this repository; pin a tag for reproducible builds:

```bash
pip install "git+https://github.com/dan-ince-aai/assemblyai-agents-python.git@v0.2.0"
uv add "assemblyai-agents @ git+https://github.com/dan-ince-aai/assemblyai-agents-python.git@v0.2.0"
pip install "assemblyai-agents[audio] @ git+https://github.com/dan-ince-aai/assemblyai-agents-python.git@v0.2.0"
```

```bash
python -c "import assemblyai_agents; print(assemblyai_agents.__version__)"
```

## Hosting

The platform has to reach your process over public HTTPS, and it resolves every
URL in DNS when the agent is created — so an address has to exist before you
deploy. Nothing in the SDK starts a tunnel: the address is yours, from wherever
you run.

| Where | Address | Run |
| --- | --- | --- |
| Railway, Render, Fly, Cloud Run, a VM | the host's URL, in `PUBLIC_BASE_URL` | `agent.serve()` (reads `PORT`) |
| Modal, Lambda, anything ASGI | the deployed function's URL | `serving.asgi(agent)` |
| Your laptop | a tunnel (ngrok, cloudflared) | `agent.serve(public_url=tunnel_url)` |

```bash
export PUBLIC_BASE_URL=https://agent.example.com   # or: RENDER_EXTERNAL_URL, https://$RAILWAY_PUBLIC_DOMAIN
export ASSEMBLYAI_API_KEY=...
export AGENT_SECRET=...                            # optional; minted per run if unset
python agent.py
```

```python
# Modal
@app.function(image=image, secrets=[modal.Secret.from_name("assemblyai")])
@modal.asgi_app()
def web():
    return serving.asgi(agent, secret=os.environ["AGENT_SECRET"])
```

The [examples repository](https://github.com/dan-ince-aai/assemblyai-agents-examples)
has a runnable file per host, including the ngrok helper for development.

## Authentication and regions

```python
from assemblyai_agents import Client, AsyncClient

client = Client()                                        # reads ASSEMBLYAI_API_KEY
client = Client(base_url="https://agents.us.assemblyai.com")
```

The default host is `https://agents.assemblyai.com`; `https://agents.us.assemblyai.com`
is the US deployment. Agents, numbers and sessions are stored per host and ids
do not cross, so keep an agent and everything attached to it on one of them.
`deploy()` keeps its id file per host for the same reason.

## Declaring tools

`@tool` turns a function into a `Tool`: the handler plus the wire definition.
Bare, the tool is **hosted by this process** — `serve()` answers it and the
deploy points the platform at it. Pass `url=` to have the platform call a
service you already run instead.

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

@tool(url="https://api.example.com/weather", http_method=HttpMethod.GET)
def weather(city: str) -> dict:
    """Current weather for a city, from a service that already exists."""
    ...
```

Checked at decoration, each with a `ConfigurationError` naming the rule:

- **Name.** `snake_case`; not `aai_credit_card_luhn_check` or
  `aai_pre_connect_context` (platform tools).
- **Description.** The docstring's first paragraph — what the model reads to
  decide whether to call it. Required. `Args:` gives per-parameter text.
- **Parameters.** Every one type-hinted: `str`, `int`, `float`, `bool`,
  `list[T]`, `dict[str, T]`, `Literal[...]`, `Enum`, `Optional[T]` / `T | None`,
  pydantic `BaseModel`. A default makes it optional. No `*args` / `**kwargs`.
- **Return type.** Required and JSON-serialisable.
- **Options.** `timeout_seconds` (1–300, default 120: lower it for anything a
  caller waits through), `response_instructions`, `dtmf_collected_arguments`
  (keypad input, phone only), `execution_mode` (`interactive` only in v1),
  `url=` / `http_method=` / `headers=` for an external service.

Calling a `Tool` calls the function unchanged, so tools stay directly testable.

### How the platform calls a tool

`POST`/`PUT`/`PATCH` tools receive the arguments as a JSON body; `GET`/`DELETE`
as query parameters (`serve()` restores numbers and booleans from the schema).
The response is stringified for the model. A non-2xx or a timeout tells the
model the tool failed — the caller hears an apology and **your client sees no
error**, so read your server log and the session's `timeline` artifact.

The platform **refuses a tool call carrying a value the conversation never
established**. Take the caller's words as arguments and do the reading in the
handler (`replies.digits_said()` helps); a value an earlier tool returned is
also accepted.

## Your own replies

Set `reply=` and the platform asks your function what to say on every turn.
The `replies` module reads the request and streams the answer:

```python
from assemblyai_agents.replies import Turn, call_tool, say, silence

def decide(turn: Turn):
    if turn.pending and turn.pending.name == "verify_caller":
        return say("Thanks, how can I help?") if turn.pending.get("verified") else say("Try again?")
    if not turn.caller_said:
        return say("Could you give me your full name?")
    return call_tool("verify_caller", caller_said=turn.caller_said)
```

`turn.pending` is the newest tool result nothing has been said about — the cue
to speak. `turn.result_of(name)` reads an earlier result back rather than
calling again. `turn.preconnect` is what a phone call's pre-connect lookup
captured. `silence()` is how a finished call ends; an agent cannot hang up.

The contract `replies` handles for you: every request streams (SSE, always);
about ten seconds to answer; the tool message is not the last one after a tool
runs; refused calls come back as prose, not JSON; there is no session id on the
request, so state is read out of the transcript. Call any model you like from
inside `decide` — the [LLM Gateway](https://www.assemblyai.com/docs/llm-gateway)
takes the same key.

## Pre-connect

Look the caller up before a phone call is answered. Give the request a
`handler=` and this process serves it at `/pre-connect/{name}`:

```python
from assemblyai_agents import Captured, PreConnectRequest

def lookup(payload: dict) -> dict:
    patient = crm.find_by_phone(payload.get("from_number", ""))
    if patient is None:
        return {"matched": False}
    return {"matched": True, "reference": patient.reference,
            "greeting": f"Welcome back, {patient.first_name}. How can I help?"}

agent = VoiceAgent(
    ...,
    pre_connect=[PreConnectRequest(
        handler=lookup,
        returns=[Captured(name="reference", path="reference")],
        allow_overrides=True,          # the response's top-level `greeting` replaces the greeting
        timeout_ms=800,
    )],
)
```

Pre-connect is telephony-only, fails open (a timeout or error means the call
proceeds without the values), and has an 800 ms ceiling. A top-level
`"reject": true` aborts the call. What it captured reaches the model — and
`turn.preconnect` — as the `aai_pre_connect_context` tool result. Up to two
requests; pass `url=` instead of `handler=` for a service you already run.

## Phone calls

```python
from assemblyai_agents import Client
from assemblyai_agents.models.rest import NumberType, PurchaseAvailablePhoneNumberRequest

number = Client().phone_numbers.purchase_available(       # billable
    PurchaseAvailablePhoneNumberRequest(country_code="US", number_type=NumberType.local,
                                        area_code=415, agent_id=agent_id)
)
```

Bring your own number by pointing your carrier's SIP trunk at AssemblyAI, then
`import_()` and `assign_agent()`. Hand a live call to a human with
`transfer_targets=[HumanTransfer(...)]` (needs `outbound_trunk_id`). Place a
call with `client.calls.create(CreateCallRequest(from_number, to_number))`.

Everything on the declaration works on a phone call and a WebSocket session
alike, because every tool is served over HTTPS by this process.

## Webhooks and sessions

```python
agent.serve(webhook_secret=WEBHOOK_SECRET, on_event=lambda e: print(e["event"]))
```

`POST /webhooks/voice-agents` is verified with `assemblyai_agents.verify()` over
the raw body before `on_event` runs. Subscribe with `client.webhooks.create(...)`.
Every session leaves a recording, a turn-by-turn timeline and metadata in
`client.sessions.get(id).artifacts`.

## Testing

```python
from assemblyai_agents.testing import create_tool_context, get_tool

ctx = create_tool_context(secrets={"orders_api_key": "test"})
ctx.http.stub("GET", "https://api.example.com/orders/W004", json={"status": "shipped"})
await get_tool(agent, "lookup_order").invoke(context=ctx, order_id="W004")
```

`agent.hosted_at("https://test.invalid", secret="k").to_request()` is the exact
payload the deploy would send, with no network. The examples' starter kit
rehearses whole calls offline through the same loop the platform runs.

## Talking to an agent from a terminal

```python
from assemblyai_agents import AgentConnection

async with AgentConnection(agent_id="agent_...") as conn:     # needs the [audio] extra
    conn.on_agent_transcript(print)
    await conn.run()
```

A test client — microphone in, speaker out. Nothing about the agent is
configured here and no tool runs here.

## Upgrading from 0.1

| Was | Now |
| --- | --- |
| `assemblyai_agents.byo` | `assemblyai_agents.replies` (alias kept, deprecated) |
| `llm=LlmConfigRequest(base_url, model, api_key)` | `reply=decide` — derived from the address (alias kept, deprecated) |
| `@tool(http=PlaintextHttpToolConfig(...))` | bare `@tool` to host here; `@tool(url=...)` for a service you run (alias kept, deprecated) |
| `tool.hosted_at(...)` in a `build(base_url)` | `agent.serve()` / `agent.deploy(public_url=...)` bind everything |
| `serve(agent, reply=, tool_secret=, llm_key=, pre_connect=)` | `agent.serve(secret=...)`; `reply` and handlers come off the declaration (aliases kept, deprecated) |
| `PreConnectRequest(url=...)` + `serve(pre_connect={path: fn})` | `PreConnectRequest(handler=fn)` |
| `AgentConnection(tools=...)`, `conn.tool()`, `ToolRouter` | removed — every tool is served over HTTPS |
| `agent.client_resident_tool_names()`, `assign_agent(agent=)` | removed / no-op |

## Documentation

Full docs, with a page per module, at the [AssemblyAI docs](https://www.assemblyai.com/docs/voice-agents/voice-agent-sdk).
