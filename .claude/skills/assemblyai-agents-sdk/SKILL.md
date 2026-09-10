---
name: assemblyai-agents-sdk
description: Build, deploy and operate AssemblyAI voice agents in Python with the assemblyai-agents SDK, the backend SDK for the Voice Agents API. Use this whenever the user wants a Python voice agent, phone agent, IVR, receptionist, order-line or any call-handling bot; wants to add, host or test tools for one; needs pre-connect (caller lookup) requests, Voice Agents webhooks, phone numbers, outbound calls or human transfers; or mentions assemblyai_agents, VoiceAgent, @tool, AgentConnection or "agents.assemblyai.com". Trigger even when the user only describes the agent's job in plain language ("make a bot that takes reservations") and never names the SDK.
---

# Building voice agents with `assemblyai-agents`

This is the **backend** SDK for the Voice Agents API. The platform owns the
call: speech to text, text to speech, turn taking, telephony. You own what the
agent knows and can do, and optionally what it says. The two meet over HTTPS,
and only over HTTPS, because a phone or SIP call has no client on the line for
the platform to ask.

So the deliverable is almost always a script that serves the user's own
functions, plus a declaration that points the platform at it. Not a client app,
and not a hand-written web service.

`references/sdk-reference.md` is the full API surface. Read it for a signature
or an exact field name; this file covers the shape of the work.

## Pick a shape first

Four, in order of how much you take on. Copy the example, do not assemble from
scratch: each one already handles the platform traps listed further down.

| The user wants | Shape | Copy |
| --- | --- | --- |
| an agent that can look things up and act | tools only, platform's model talks | `examples/tools_only_agent.py` |
| control over what is said | your own replies (`llm=`) | `examples/one_file_agent.py` |
| stages, or cost control, or a stage that provably cannot do certain things | subagent routing | `examples/subagents.py` |
| a project rather than a script: tests, a system of record, a call flow to grow | the full kit | `examples/starter/` |

Start at the top of that table and move down only for a stated reason. Most
first agents are the first row, and the first three are a single file each.

The one-file shapes all do the same three things in this order, and the order
matters because tool URLs have to exist before the agent is created:

```python
with public_address(PORT) as base_url:      # examples/expose.py: ngrok
    agent = build(base_url)                 # tool URLs point at this process
    agent_id = deploy(agent)                # create, or update a stored id
    serve(agent, reply=decide, port=PORT)   # blocks; the platform calls in
```

`examples/starter/` is the same three steps in `run.py`, around a project with
`store.py` (the system of record), `agent.py` (tools and declaration),
`reply.py` (the call flow), `flow.py` (stage machinery), `rehearse.py` (a whole
call offline) and tests. Change those four files in that order.

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

4. **Serve it** with `serving.serve(agent, reply=..., tool_secret=..., ...)`,
   which answers every route the platform will call straight off the
   declaration. Do not hand-write a web service for this; if the project
   already has one, mount `routes(agent, ...)` into it instead. The address has
   to be public HTTPS: the API resolves every tool and pre-connect hostname in
   public DNS at create/update time (`ValidationError: … URL host … does not
   resolve`), so a placeholder cannot be deployed. `examples/expose.py` starts
   ngrok and yields the address, unless `PUBLIC_BASE_URL` is already set. Keep
   it running — a dead origin surfaces only as the model apologising to the
   caller.

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
   - Run the process and call the number. Every tool call and every reply
     arrives there over HTTPS, and the log shows the exact body and headers the
     platform sent, so there is nothing separate to run.
   - Rehearse the whole call offline: run the reply engine and the tools in a
     loop with no platform at all, which is what `examples/starter/rehearse.py`
     does and what its tests drive. Seconds per change, and it belongs in CI.
   - A real call: `AgentConnection` from a terminal (needs the `[audio]` extra
     and PortAudio), or a phone number.
   - When a hosted tool fails, the caller only hears an apology and the client
     sees no error: read the server/tunnel logs and
     `client.sessions.get(session_id).artifacts` (the `timeline` artifact lists
     each turn with its `trigger`, e.g. `tool_result`).

7. **Attach a phone number** only after every tool has `http=`: the SDK refuses
   `assign_agent(..., agent=agent)` for a declaration with client-resident
   tools, because a phone call has no connected client to run them.

## Running it: one command, your code, an address

Everything the platform needs arrives over HTTPS, so the shape is: get an
address, deploy the declaration built against it, then serve. In that order,
because the tool URLs have to be known before the agent is created.

`assemblyai_agents.serving.serve(agent, reply=decide, tool_secret=..., llm_key=...)`
is the serving half. It answers from the declaration — every `@tool` on it, the
reply function, pre-connect handlers, webhooks, `/healthz` — on the standard
library alone, so there is no framework to choose and no service to write.
`routes()` returns the same handlers as plain callables for a project that
already has an application.

`examples/expose.py` is the address half: it starts ngrok unless
`PUBLIC_BASE_URL` is already set. It is a script in the examples and not part
of the SDK, because it is a workaround until agent code can be deployed
directly, and nothing in `assemblyai_agents` knows a tunnel exists.

Put together, a whole agent is one command:

```python
with public_address(PORT) as base_url:
    os.environ["PUBLIC_BASE_URL"] = base_url   # before the declaration is built
    agent = build(base_url)
    agent_id = deploy(agent)
    serve(agent, reply=decide, port=PORT, tool_secret=SECRET, llm_key=SECRET)
```

`examples/tools_only_agent.py` is that in a single file, with the platform's
own model running the conversation and the script answering only tool calls.
`examples/one_file_agent.py` is the same with `llm=` added, so the script
decides every reply too. `examples/starter/run.py` is the same three steps for
a bigger project. Point a phone number at the agent id any of them prints and a
real caller takes the identical path.

Start a user on tools only. Move them to their own replies when a prompt is not
a strong enough guarantee for what has to be said: on a test call the tools-only
agent looked a value up correctly and then answered a different question, which
is fine for a shop and not fine for a disclosure.

## Subagents, when one prompt is doing too much

`examples/subagents.py` routes a call between stages, each with its own model,
prompt and allowed tools: a cheap model to check who is on the line, a stronger
one for the conversation, the strongest only when it goes wrong. Reach for it
when the user describes stages, or wants to control cost, or wants a stage that
provably cannot do certain things.

Four things to get right, all of them learned by getting them wrong:

- **Handing over is not a config change.** With `llm=` pointing at their code,
  a handover is choosing a different model and prompt for the next turn.
  `session.update` exists but is a WebSocket message, so it is unavailable on a
  phone call.
- **The allowlist is enforced in their code**, because the platform runs
  whatever tool call it is handed. Offer each stage only its own tool schemas
  *and* refuse an out-of-scope call if one comes back.
- **A subagent prompt replaces the platform's**, and with it the spoken-output
  guidance the platform normally appends. Put those rules in every subagent
  prompt or a model will emit markdown into speech.
- **Do not forward one model's tool history to another** that was not given
  those tools; it is rejected. Flatten prior calls and results into plain
  notes. Also strip trailing whitespace from assistant lines and make sure the
  message list ends on a user turn, both of which a gateway will refuse.

Two habits that follow from this:

- Print and flush from a serving process. A log that only appears when the
  process exits is no use while a call is in progress.
- Check a change without picking up the phone by rehearsing the call offline:
  run the reply logic and the tools in the platform's own loop with no network,
  as `examples/starter/rehearse.py` does and its tests drive. Milliseconds per
  call, and it belongs in CI.

## Where a tool runs

| Tool declared… | Who runs it | Works on | Use when |
| --- | --- | --- | --- |
| with `http=PlaintextHttpToolConfig(url=..., http_method=..., headers=[...])` | whatever is at that address, which can be a script on the developer's own machine | phone, SIP and WebSocket | **almost always** |
| without `http=` (client-resident) | the process holding the WebSocket, via `AgentConnection(tools={name: fn})` | WebSocket only | a desktop or browser session that needs local state, and nothing else |

**Use `http=` unless there is a specific reason not to.** A phone or SIP call
has no connected client, so a client-resident tool cannot be answered on one,
and the SDK refuses to attach a number to an agent that still has one. Reaching
for client-resident tools to avoid standing something up is a false economy:
the agent then only works from a browser.

Serving them does not mean writing a backend.
`assemblyai_agents.serving.serve(agent, reply=decide, tool_secret=..., llm_key=...)`
answers the platform from the declaration, on the standard library alone, so a
whole agent is one script: see `examples/one_file_agent.py`, which declares its
tools, decides what to say, deploys itself, gets an address and serves, in that
order. `routes()` from the same module returns the identical handlers as plain
callables for anyone who would rather host them in their own application.

The public address comes from `PUBLIC_BASE_URL`, or from `examples/expose.py`,
which starts ngrok or cloudflared. That helper is in the examples rather than
the SDK because it is a workaround until agent code can be deployed directly;
when that ships it is the only piece that changes.

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

A handy pattern is a `hosted(path)` helper that builds the config from
`PUBLIC_BASE_URL`, so one declaration works against a tunnel, a staging host or
a deployment without editing:

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

- **A handler will be called with a placeholder sooner or later.** A model
  that has not been told the value asks for it anyway: a live call reached
  `find_policy(policy_number="policy number")`, the parameter's own description
  echoed back as its value. Validate in the handler and return a refusal the
  model can read (`{"found": false, "ask": "read me the policy number"}`),
  rather than treating the string as data. `serving.Refused(message)` does the
  same for anything that should not have been called at all. The platform's own
  guard catches some of these on a `llm=` agent, but it is not a substitute:
  the handler is the only place that knows what a real value looks like.

Prompt guidance for `system_prompt`: spoken output, so short sentences, no
markdown, no lists; state when to call each tool by name and, for each argument,
that it comes from the caller and is never to be guessed; tell it what to do
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
it rather than hand-rolling SSE and transcript parsing. It is deliberately
small: reading a request, three ways to answer, and the streaming shape.

```python
from assemblyai_agents.byo import Turn, call_tool, say, silence, stream

def decide(turn):
    if turn.pending and turn.pending.name == "verify_caller":
        return say("Thanks, how can I help?") if turn.pending.get("verified") else say("Try again?")
    if not turn.caller_said:
        return say("Could you give me your full name?")
    return call_tool("verify_caller", caller_said=turn.caller_said)

# on POST /v1/chat/completions, in whatever framework the project uses:
turn = Turn.from_request(body)
return StreamingResponse(stream(turn, decide(turn)), media_type="text/event-stream")
```

What it gives you, each of which is a trap from the list above:

| | |
| --- | --- |
| `Turn.from_request(body)` | the request, read: `caller_said`, `spoken`, `tool_names`, `preconnect`, `pending`, `results` |
| `turn.pending` | the tool result nothing has been said about yet, which is the cue to speak; `None` once something has |
| `turn.pending.ran` | `False` when the platform refused or failed the call, so a refusal is never reported as a result |
| `turn.preconnect` | the pre-connect captures, read out of the `aai_pre_connect_context` result |
| `turn.result_of(name)` | an earlier result, to read back rather than call again |
| `turn.answer_following(fragment)` | the caller's reply to a question you asked, for a value collected over turns |
| `call_tool(name, **args)` | drops arguments the call has not established, so the platform accepts it |
| `say(text)` / `silence()` | words, or nothing, which is how a finished call ends |
| `stream(turn, answer)` / `json_body(turn, answer)` | the streaming shape, and the plain one for curl |
| `digits_said(text)` | digits out of "four four seven one", "forty one eleven", "double one" |

How `decide` is organised is not the SDK's business. The starter shows one way,
`Stages` and `Memo` in its own `flow.py`, forty lines you can read and change.
Reach for that shape when a call has scripted ends and a conversational middle;
a plain function with a few branches is fine for anything smaller.

Nothing in the SDK knows what a tunnel is. An agent's tool URLs come from
`PUBLIC_BASE_URL` and it does not care what set the value: a tunnel today, a
staging host, a deployment later. `examples/expose.py` is the only file that
starts one.

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

A working endpoint, tools and replies in one process, is
`examples/one_file_agent.py`. If the project already has a web application,
`routes(agent, reply=decide, ...)` returns the same handlers as plain
callables to mount into it. On a call the platform arrives three times per
exchange: to ask what to say, to run the tool that
answer asked for, and again with the result.

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
- A rehearsal of each call worth caring about, offline, plus one real call once a number is attached.
