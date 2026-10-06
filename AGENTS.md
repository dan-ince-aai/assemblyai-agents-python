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
  directly. The one shape it does recognise is the address AssemblyAI mints for
  a hosted service (`_urls.py`), and only to refuse it: the API refuses that
  write too, so this is the rule above, not an exception to this one.
- **A deployment's own address is never shown to a customer.** It names one
  deployment and dies with the next, nothing an agent stores may carry it, and
  every caller resolves it per call from the agent ID. So no command prints it
  and no docstring teaches it; the path form is what goes on the agent.
- **`assemblyai_agents/models/` is generated, not written.** `rest.py` and
  `ws.py` come out of datamodel-codegen against the API's OpenAPI document, so
  a new wire field arrives by regenerating them from the current spec rather
  than by hand. Anything hand-written about that field — a `VoiceAgent` field,
  a CLI flag, a line of README — sits outside `models/`.
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

**A tag that introduces a new wire field waits until that field is deployed,
not until it is merged.** Those are different states, and the gap between them
is real: the field that prompted this rule merged to the platform's master and
was never deployed at all.

Shipping into that gap fails silently rather than loudly. The config structs on
the other side do not forbid unknown keys, and validation only reads keys it
knows, so a field the deployed server has not got is accepted and ignored. No
error reaches anyone. A setting that does nothing is indistinguishable from a
setting that works, which is the worst way for this to go wrong and the reason
the rule is written down rather than assumed.

Nothing here enforces it. CI runs the tests and an install check; there is no
release workflow, no tag trigger and no version gate, so the person tagging is
the check. The package has no capability negotiation with the API either, so it
cannot ask what the server supports and refuse.

What it can do is make the mismatch visible afterwards. Every field on the
session-config models in `models/` is echoed back in `session.ready`, so a
caller who compares what they set against what comes back can see a server that
dropped it. That only works while those models carry the field — so regenerate
`models/` for a new wire field in the same change that adds the field, not
later.
