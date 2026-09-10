"""Create the agent on first run, update it after that.

    export ASSEMBLYAI_API_KEY=... PUBLIC_BASE_URL=https://<host> TOOL_SECRET=... LLM_API_KEY=...
    BYO_LLM=1 python deploy.py         create, or update the stored id
    python deploy.py --show            read the deployed agent back
    python deploy.py --delete          remove it and forget the id

The id is kept in .agent_id next to this file, per host, so re-running never
leaves a second copy behind. Update sends the whole declaration, because the
API replaces the stored agent rather than merging into it.
"""

import argparse
import os
import sys
from pathlib import Path

from assemblyai_agents import Client, NotFoundError

from agent import agent

BASE_URL = os.environ.get("AAI_BASE_URL", "https://agents.assemblyai.com").rstrip("/")
# Agents live per host and ids do not cross, so the note of which one is kept
# per host too.
SUFFIX = "" if "agents.assemblyai.com" == BASE_URL.split("//")[-1] else "." + BASE_URL.split("//")[-1].split(".")[0]
ID_FILE = Path(__file__).with_name(f".agent_id{SUFFIX}")


def stored_id() -> str | None:
    if os.environ.get("AGENT_ID"):
        return os.environ["AGENT_ID"]
    return ID_FILE.read_text().strip() or None if ID_FILE.exists() else None


def describe(deployed) -> str:
    lines = [f"{deployed.id}  {deployed.name!r}  voice={deployed.voice.voice_id}"]
    for tool in deployed.tools or []:
        keypad = ", ".join(p.parameter_name for p in tool.dtmf_collected_arguments or [])
        lines.append(
            f"  {tool.name:22} {tool.timeout_seconds:>4}s  "
            f"{tool.http.url if tool.http else 'client-resident'}"
            + (f"  keypad: {keypad}" if keypad else "")
        )
    for entry in deployed.pre_connect_requests or []:
        lines.append(f"  pre-connect  {entry.http.url}  overrides={entry.allow_overrides}")
    for llm in deployed.llm or []:
        lines.append(f"  replies from {llm.base_url}  model={llm.model}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--delete", action="store_true")
    args = parser.parse_args()

    client = Client(base_url=BASE_URL)
    agent_id = stored_id()
    print(f"host {BASE_URL}  (id file {ID_FILE.name})")

    if args.show:
        if not agent_id:
            print("nothing deployed yet")
            return 1
        print(describe(client.agents.get(agent_id)))
        return 0

    if args.delete:
        if not agent_id:
            print("nothing deployed yet")
            return 0
        try:
            client.agents.delete(agent_id)
            print(f"deleted {agent_id}")
        except NotFoundError:
            print(f"{agent_id} was already gone")
        ID_FILE.unlink(missing_ok=True)
        return 0

    if not os.environ.get("PUBLIC_BASE_URL"):
        sys.exit(
            "PUBLIC_BASE_URL is unset, so every tool would be client-resident and could not "
            "answer a phone call. Expose backend.py over HTTPS and set it."
        )

    if agent_id:
        try:
            deployed = client.agents.update(agent_id, agent)
            print(f"updated {deployed.id}")
        except NotFoundError:
            deployed = client.agents.create(agent)
            print(f"the stored agent was gone; created {deployed.id}")
    else:
        deployed = client.agents.create(agent)
        print(f"created {deployed.id}")

    ID_FILE.write_text(deployed.id)
    print(describe(client.agents.get(deployed.id)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
