"""Talk to the deployed agent from a terminal, with a scripted caller.

    export ASSEMBLYAI_API_KEY=...
    python drive.py                 the happy path
    python drive.py wrong_name      verification that fails

The work is `scripted_call` in the SDK: it opens a real session with device
audio off, sends each line as a user turn, and hands back the transcript. The
same scenarios drive the offline rehearsal, so a call that works in rehearse.py
should work here.
"""

import os
import sys
from pathlib import Path

from assemblyai_agents.drive import call

from rehearse import SCENARIOS


def agent_id() -> str:
    if os.environ.get("AGENT_ID"):
        return os.environ["AGENT_ID"]
    for name in (".agent_id", ".agent_id.us"):
        path = Path(__file__).with_name(name)
        if path.exists() and path.read_text().strip():
            return path.read_text().strip()
    sys.exit("no agent id: run deploy.py first, or set AGENT_ID")


if __name__ == "__main__":
    scenario = sys.argv[1] if len(sys.argv) > 1 else "happy"
    if scenario not in SCENARIOS:
        sys.exit(f"unknown scenario {scenario!r}; pick from {', '.join(SCENARIOS)}")
    lines, _ = SCENARIOS[scenario]

    transcript = call(
        agent_id(),
        lines,
        on_turn=lambda speaker, text: print(f"{'AGENT ' if speaker == 'agent' else 'CALLER'}: {text}\n"),
    )
    print(f"session {transcript.session_id}")
    sys.exit(0 if transcript.ok else 1)
