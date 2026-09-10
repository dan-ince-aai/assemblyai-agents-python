# Building a voice agent with this SDK

Read this before writing code. The full guidance is
`.claude/skills/assemblyai-agents-sdk/SKILL.md` (workflow, pitfalls, telephony,
bring-your-own-LLM) with the API surface in
`.claude/skills/assemblyai-agents-sdk/references/sdk-reference.md`. Claude Code
loads those automatically; every other coding agent should open them now. This
file is the short version.

## What you are building

The platform owns the call: speech to text, text to speech, turn taking,
telephony. You own what the agent knows and can do, and optionally what it
says. The two meet over HTTPS, and only over HTTPS, because a phone or SIP call
has no client on the line for the platform to ask.

So the deliverable is a script that serves the user's own functions, plus a
declaration pointing the platform at it. Not a client app, and not a
hand-written web service.

## Pick a shape

Copy one, do not assemble from scratch. Each already handles the traps below.

| The user wants | Shape | Copy |
| --- | --- | --- |
| an agent that can look things up and act | tools only, platform's model talks | `examples/tools_only_agent.py` |
| control over what is said | own replies (`llm=`) | `examples/one_file_agent.py` |
| stages, cost control, or a stage that provably cannot do certain things | subagent routing | `examples/subagents.py` |
| a project rather than a script | the full kit | `examples/starter/` |

Start at the top and move down only for a stated reason. The first three are a
single file each.

`pip install` does not give you the examples. Clone the repo to copy one, and
copy `expose.py` alongside it: every shape imports `public_address` from that
file and it is not part of the package. `examples/starter/` ships its own copy.

## The shape of every one of them

```python
with public_address(PORT) as base_url:      # examples/expose.py: starts ngrok
    agent = build(base_url)                 # tool URLs point back at this process
    agent_id = deploy(agent)                # create, or update a stored id
    serve(agent, reply=decide, port=PORT)   # blocks; the platform calls in
```

The order is not stylistic. The API resolves every tool hostname in public DNS
when the agent is created, so the address has to exist first.

`serve()` answers every route the platform will call, read off the declaration,
on the standard library alone. Do not write a web service. If the project
already has one, mount `routes(agent, ...)` into it.

## Rules that cost a call when broken

- **Give every tool `http=`.** A tool without it is resolved by the connected
  WebSocket client, so it cannot run on a phone call, and the SDK refuses to
  attach a number to an agent that still has one.
- **Never send an argument value the conversation has not established.** The
  platform refuses a tool call carrying an invented or reworded value, and the
  refusal reaches the caller as silence. Pass what was actually said, verbatim,
  and drop empty values (`byo.established(**arguments)` does that).
- **Validate arguments in the handler.** A model that was never told a value
  asks for it anyway: a live call reached `find_policy(policy_number="policy
  number")`, the parameter's description echoed back as its value. Return a
  refusal the model can read rather than treating the string as data.
- **A tool result may be prose, not JSON.** "The caller did not finish entering
  card_number (too_short), so the tool was not called" is a real result.
  Reporting that to the caller as a failure of the thing they were doing is a
  lie. Distinguish did-not-run from ran-and-failed.
- **There is no session id on a reply request.** The body carries the whole
  transcript and nothing naming the call, so per-call state cannot be a dict
  keyed by session. Read state back out of the transcript, and for the one
  thing it cannot tell you — what the agent already said, since a line the
  caller talked over never comes back — keep a note keyed on a value the call
  established.
- **A subagent prompt replaces the platform's**, including the spoken-output
  guidance it normally appends. Without those rules re-stated, a model emits
  markdown into speech.
- **Do not forward one model's tool history to another** that was not given
  those tools; the gateway rejects it. Flatten prior calls and results into
  plain notes.
- **Print and flush from a serving process.** A log that only appears at exit
  is no use while a call is in progress.
- **Never write an API key into source.** The client reads
  `ASSEMBLYAI_API_KEY`. If it is unset, ask where it lives.

## Checking work without picking up the phone

- `agent.to_request()` — a `ConfigurationError` names the exact rule broken,
  with no network call. (Avoid a full `model_dump()` in logs; it prints tool
  header secrets.)
- `assemblyai_agents.testing` — tools are plain callables, unit-test them.
- `examples/starter/rehearse.py` — runs the reply logic and the tools in the
  platform's own loop, offline, in milliseconds. Its tests are whole calls, and
  they belong in CI.

Then call the number. Every tool call and every reply arrives in your terminal
over HTTPS, which is the same path a browser, a phone and SIP all take.

## Working on the SDK itself

`assemblyai_agents/` is deliberately small: it declares agents, validates them
before the round trip, and serves the platform's requests. Opinions about how
to run a conversation belong in `examples/`, not in the package. `ngrok` is a
workaround until agent code can be deployed directly, so it lives in
`examples/expose.py` and nothing in the package knows a tunnel exists.

```bash
uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python -m pytest -q
```
