# Working on this agent

`README.md` explains what each file is and how to run it. This file is for a
coding agent picking the project up.

## Change these four files, in this order

1. `store.py` — the system of record. It is mocked. Replace it with the real
   services: a database, an API, a vector store, whatever the business already
   has. Everything else reads through it, so nothing above needs to know.
2. `agent.py` — the tools and the declaration. One `@tool` function per thing
   the agent can do, plus the prompt, voice and greeting. Every tool needs
   `http=`; see the rules below.
3. `reply.py` — what the agent says, as stages. This is the call flow.
4. `model.py` — the handful of turns a script cannot decide. Keep it small; a
   model call per turn is slower and less predictable than a rule.

`backend.py`, `run.py`, `flow.py` and `deploy.py` rarely need changing.
`expose.py` is a tunnel, and is temporary: point `PUBLIC_BASE_URL` at a staging
host or a deployment and it is never started.

## After every change

```bash
.venv/bin/python rehearse.py happy      # a whole call, offline, milliseconds
.venv/bin/python -m pytest -q           # whole calls, asserted
```

`rehearse.py` runs the platform's own loop with no network: ask the reply
engine what to say, run any tool it asks for, hand the result back, ask again.
Use it instead of calling the number. The tests run with the model off, so the
fallback wording stays honest.

## Rules that cost a call when broken

- **Every tool needs `http=`.** A tool without it is answered by a connected
  WebSocket client, so it cannot run on a phone call at all.
- **Never send an argument value the conversation has not established.** The
  platform refuses a tool call carrying an invented or reworded value, and the
  refusal reaches the caller as silence. Pass what was said, verbatim, and drop
  empty values with `byo.established(**arguments)`.
- **A tool result may be prose, not JSON** — "the caller did not finish
  entering card_number (too_short), so the tool was not called" is a real one.
  Do not report that to the caller as a failure of what they were doing.
- **Restate the spoken-output rules in any prompt you write.** A prompt you
  supply replaces the platform's, including the guidance that stops a model
  reading markdown aloud.
- **Print and flush.** A log that appears only at exit is no use during a call.
- **Never write an API key into source.** The client reads
  `ASSEMBLYAI_API_KEY`.

The SDK's full guidance, if this project is still inside the SDK repo, is
`.claude/skills/assemblyai-agents-sdk/SKILL.md` and its `references/`. Otherwise
see https://github.com/dan-ince-aai/assemblyai-agents-python.
