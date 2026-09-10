"""Create the Pizza Line agent, or update it if AGENT_ID is already set.

    export ASSEMBLYAI_API_KEY=...
    python examples/deploy.py                 # creates, prints the new id
    AGENT_ID=agent_... python examples/deploy.py   # replaces that agent's config

``agents.update`` sends the whole declaration: PUT replaces the stored agent
rather than merging into it, so keep the declaration in one place (pizza_line.py)
and redeploy from there.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from assemblyai_agents import Client
from pizza_line import agent

client = Client()  # reads ASSEMBLYAI_API_KEY

agent_id = os.environ.get("AGENT_ID")
if agent_id:
    deployed = client.agents.update(agent_id, agent)
    print(f"updated {deployed.id}")
else:
    deployed = client.agents.create(agent)
    print(f"created {deployed.id}")

print("tools:", [t.name for t in deployed.tools or []])
print(f"\nexport AGENT_ID={deployed.id}")
