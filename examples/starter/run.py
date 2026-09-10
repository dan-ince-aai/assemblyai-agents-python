"""Run this agent: address, deploy, serve. One process, one command.

    export ASSEMBLYAI_API_KEY=...
    python run.py

In order, this gets a public address for this machine, builds the declaration
against that address, deploys it, and then answers the platform's requests from
the functions in this project until you stop it. Point a phone number at the
agent id it prints and a real caller arrives down the same path.

The only part that knows a tunnel exists is `expose.py`, in this directory.
Everything else reads `PUBLIC_BASE_URL`, so setting that to a staging host or a
deployment makes the tunnel disappear with no other change, and deletes that
file when agent code can be deployed directly.
"""

import os
import sys
from pathlib import Path

# This project first, so its modules win over anything of the same name that
# happens to be on the path already.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from expose import public_address

PORT = int(os.environ.get("PORT", "8000"))


def main() -> int:
    if not os.environ.get("ASSEMBLYAI_API_KEY"):
        sys.exit("set ASSEMBLYAI_API_KEY")
    os.environ.setdefault("TOOL_SECRET", "starter-tool-secret")
    os.environ.setdefault("LLM_API_KEY", "starter-llm-key")
    os.environ.setdefault("BYO_LLM", "1")

    with public_address(PORT) as base_url:
        # Set before importing anything that builds the declaration, because
        # the tool URLs are read from here at import time.
        os.environ["PUBLIC_BASE_URL"] = base_url

        from assemblyai_agents.serving import serve

        import backend
        import reply
        from agent import agent
        from deploy import main as deploy

        if deploy() != 0:
            return 1
        print(f"\nserving from {base_url}; Ctrl-C to stop\n", flush=True)
        serve(
            agent,
            reply=reply.decide,
            port=PORT,
            tool_secret=os.environ["TOOL_SECRET"],
            llm_key=os.environ["LLM_API_KEY"],
            pre_connect={"/pre-connect/lookup": backend.lookup},
            webhook_secret=os.environ.get("WEBHOOK_SECRET"),
            on_event=lambda event: print(f"[webhook] {event.get('event') or event.get('type')}"),
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
