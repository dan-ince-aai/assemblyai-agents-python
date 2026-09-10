"""Call the one-file agent from a script, to check it without a phone.

    AGENT_ID=agent_... python drive_one_file.py

The tools are HTTP tools, so this exercises exactly the path a phone call uses:
the platform fetches them from the running process either way.
"""

import os
import sys
from pathlib import Path

from assemblyai_agents.drive import call

SCRIPT = [
    "Hi, I want to check an order.",
    "It's one zero four two.",
    "No that's all, thanks.",
]

agent_id = os.environ.get("AGENT_ID") or (
    Path(__file__).with_name(".one_file_agent_id").read_text().strip()
    if Path(__file__).with_name(".one_file_agent_id").exists()
    else ""
)
if not agent_id:
    sys.exit("set AGENT_ID, or run one_file_agent.py first")

transcript = call(
    agent_id,
    SCRIPT,
    on_turn=lambda who, text: print(f"{'agent' if who == 'agent' else 'you  '}: {text}"),
)
print(f"\nsession {transcript.session_id}")
sys.exit(0 if transcript.ok else 1)
