# Voice agent starter

A complete, running agent you can rename and rewrite. It books dental
appointments, which is not the point: the point is the shape underneath, which
is the same for a support line, a collections line, a claims line or an order
desk.

```
store.py     the system of record, mocked        <- replace with your services
agent.py     the tools and the declaration       <- your tools, your prompt
reply.py     what the agent says, as stages      <- your call flow
flow.py      Stages and Memo, forty lines        <- the organising opinion
model.py     a model for the turns a script cannot cover
backend.py   the four surfaces the platform calls (mounted in one call)
rehearse.py  run a whole call locally, no network
drive.py     run a call against the deployed agent, no microphone
deploy.py    create, update, show, delete
tests/       whole calls, asserted
run.sh       backend, address, deploy
```

## Run it in two minutes, offline

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python \
    "git+https://github.com/dan-ince-aai/assemblyai-agents-python.git" \
    fastapi "uvicorn[standard]" pytest pytest-asyncio httpx

.venv/bin/python rehearse.py happy        # a caller who books
.venv/bin/python rehearse.py by_phone     # recognised from their number
.venv/bin/python rehearse.py wrong_name   # verification that fails
.venv/bin/python rehearse.py objection    # something the script does not cover
.venv/bin/python -m pytest -q             # 14 tests, all whole calls
```

`rehearse.py` runs the platform's own loop: ask the reply engine what to say,
run any tool it asks for, hand the result back, ask again. No network, no
account, milliseconds per call. Edit `reply.py` and run it again.

## Run it against the platform

```bash
export ASSEMBLYAI_API_KEY=...
./run.sh                              # or set PUBLIC_BASE_URL to your own address first
.venv/bin/python drive.py happy       # a scripted caller, no microphone
tail -f .run/backend.log              # every decision and every tool call
```

`run.sh` starts the backend, works out a public address, and deploys. If
`PUBLIC_BASE_URL` is already set it uses that and starts nothing; otherwise it
starts ngrok as a convenience. How your machine becomes reachable is your
choice, and nothing in the SDK has an opinion about it.

## The idea worth keeping: stages

A call has a shape. The ends are usually a script, the middle is a
conversation, and one handler trying to be both turns into a tangle of
conditions. So the reply engine is a short list of stages, each of which owns a
few turns and hands over when its own test says it is done:

```python
stages = Stages()                    # from flow.py, forty lines, yours to change

@stages.stage("identify", until=verified)
def identify(turn):
    if not turn.caller_said:
        return say(NAME_ASK)
    return call_tool("verify_caller", caller_said=turn.caller_said)

@stages.stage("notice", until=lambda turn: memo.has(call_key(turn), "notice"))
def notice(turn):
    memo.note(call_key(turn), "notice")
    return say(NOTICE)               # word for word, exactly once

@stages.stage("book", until=told_them)
def book(turn): ...                  # tools, a model, whatever it takes

@stages.stage("close")
def close(turn):
    return silence()                 # a finished call is quiet
```

`Stages` and `Memo` live in `flow.py`, not in the SDK. They are an opinion
about organising a call, and a short one you can read: the SDK only reads the
request and streams the answer.

Scripted stages never reach for the model, so their wording cannot drift. That
is the whole reason to write replies yourself: a required disclosure, a fixed
order of steps, an amount that has to come from a ledger rather than from a
sentence.

## And the model, where a script cannot help

`model.py` makes two kinds of call and no others:

- **Classify.** What did the caller actually want? A keyword list gets the
  obvious phrasings; most real callers do not use them.
- **Deliver a settled position.** The practice's answer is decided in
  `reply.POSITIONS`; the model is handed that position plus what the caller
  said, and chooses the words. It cannot change the position, and if it is
  slow or unavailable the written wording goes out instead.

Point it anywhere OpenAI-compatible:

```bash
export MODEL_BASE_URL=https://api.example.com/v1 MODEL_API_KEY=... MODEL_NAME=...
export MODEL=off      # or turn it off entirely; every test runs this way
```

## What to change, in order

1. **`store.py`** — replace each function with a call into your own systems.
   Keep the signatures; everything above them is unaffected.
2. **`agent.py`** — your tools, your prompt, your voice. Two habits worth
   keeping: take the caller's words rather than a value you derived from them,
   because the platform refuses arguments the conversation never established;
   and return the ids you were asked about, because whatever writes the reply
   sees the tool result and nothing else.
3. **`reply.py`** — your stages. Put the wording that must not drift in
   constants, and the positions you are willing to state in `POSITIONS`.
4. **`tests/test_call.py`** — one test per whole call. They run offline, so
   there is no reason not to have a lot of them.
5. **`rehearse.py`** — add a scenario for each call you care about. `drive.py`
   replays the same ones against the deployed agent.

## Environment

| Variable | What it does |
| --- | --- |
| `ASSEMBLYAI_API_KEY` | Deploys and connects |
| `PUBLIC_BASE_URL` | Public HTTPS address of `backend.py` |
| `TOOL_SECRET` | Presented by the platform on every tool and pre-connect call |
| `LLM_API_KEY` | Presented by the platform on the reply endpoint |
| `BYO_LLM=1` | Replies come from `reply.py` rather than the platform's model |
| `MODEL_BASE_URL`, `MODEL_API_KEY`, `MODEL_NAME` | The model for free-form turns |
| `MODEL=off` | Never call a model; canned wording everywhere |
| `WEBHOOK_SECRET` | Verifies signed session and call events |
| `AGENT_NAME`, `VOICE` | What the agent calls itself, and how it sounds |
| `DEMO_MATCH_ANY=1` | Treat any caller as the demo patient, for a phone demo |
| `AAI_BASE_URL` | The regional host, if not the default |

## Test data

| Reference | Patient | Phone | Notes |
| --- | --- | --- | --- |
| `4471` | Maria Delgado | +1 469 555 0142 | nothing outstanding |
| `8820` | Tomas Okonkwo | +1 972 555 0461 | £45 outstanding, mentioned after the notice |

Callers read references out in words as often as in figures, so
`digits_said("it's four four seven one")` is what turns that into `4471`.
