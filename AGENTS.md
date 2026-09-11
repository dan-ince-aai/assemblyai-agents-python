# Working on `assemblyai-agents`

This repository is the package and nothing else. Worked examples, the starter
project, the tunnel helper and the Claude Code skill live at
https://github.com/dan-ince-aai/assemblyai-agents-examples — if you are here to
*build* an agent rather than change the SDK, go there.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python -m pytest -q
```

## What belongs in here

The package declares agents, validates a declaration before the round trip, and
answers the requests the platform sends. That is the whole remit.

- **A rule the server would reject belongs in the declaration**, checked at
  import time with a `ConfigurationError` that names the rule. A round trip to
  learn a tool name is not snake_case is a round trip wasted.
- **Opinions about how to run a conversation do not belong here.** How to stage
  a call, when to reach for a model, how to organise a reply function: all of
  that is an example, not an API. `assemblyai_agents.byo` gives the pieces
  (`Turn`, `say`, `call_tool`, `stream`); it does not give a framework.
- **Nothing in the package knows a tunnel exists.** Tool URLs come from whatever
  the caller passes to `hosted_at`, and the package does not care what put the
  value there — a tunnel today, a deployment later. The ngrok helper is a script
  in the examples repository and is deleted when agent code can be deployed
  directly.
- **No new vocabulary for a wire field.** Every `VoiceAgent` field maps onto
  `AgentCreateRequest`, so `to_request()` is inspectable and assertable without
  a network call. A shortcut that relocates a field is a second vocabulary to
  learn.

## Testing

`tests/` is the contract. Every rule the declaration enforces has a test that
asserts the error names the rule, because the error message is the API for
anyone who gets it wrong.

Changes that affect how an agent is built also need checking against the
examples repository, which installs this package from `main` and proves each
example still builds a valid declaration. Its CI runs weekly for that reason. If
you change something an example depends on, open a pull request there too.

## Releasing

`pyproject.toml` holds the version; tag it to match. Consumers install from a
git URL, so a tag is the only stable reference they have.
