"""Tools only: the platform does the talking, your code does the looking up.

    export ASSEMBLYAI_API_KEY=...
    python tools_only_agent.py

The smallest useful shape. You write `@tool` functions; the platform's own
model runs the conversation and calls them. There is no reply endpoint here, so
nothing decides wording except the system prompt.

Start here. Reach for your own reply generation (`llm=`, and see
`one_file_agent.py`) when the words matter more than a prompt can guarantee: a
disclosure that has to be read exactly, a fixed order of steps, an amount that
must come from a ledger rather than from a sentence.

The difference shows up quickly. On a test call this agent looked up Saturday's
hours correctly and then answered a different question, because what it says is
the model's decision and only the prompt shapes it. That is fine for a shop and
not fine for a disclosure.

What does not change either way is this file's shape. The tools are HTTP tools,
so the platform fetches them itself and a phone call works. `expose.py` gets
this process an address, `serve()` answers from the declaration, and there is
no backend in between.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from assemblyai_agents import Client, NotFoundError, VoiceAgent, tool
from assemblyai_agents.byo import digits_said
from assemblyai_agents.models.rest import (
    HttpToolHeaderInput,
)
from assemblyai_agents.serving import claim_port, serve

from expose import public_address

PORT = int(os.environ.get("PORT", "8000"))
SECRET = os.environ.get("SHARED_SECRET", "change-me-" + os.urandom(4).hex())
ID_FILE = Path(__file__).with_name(".tools_only_agent_id")

# Stand in for whatever you would really query: a stock system, a warehouse
# API, a database. The tool surface does not change.
STOCK = {
    "8mm drill bit": {"in_stock": 14, "aisle": "4", "price": "3.40"},
    "gorilla glue": {"in_stock": 0, "aisle": "9", "price": "6.99"},
    "garden twine": {"in_stock": 62, "aisle": "12", "price": "2.10"},
}
HOURS = {
    "monday": "8 to 6", "tuesday": "8 to 6", "wednesday": "8 to 6",
    "thursday": "8 to 8", "friday": "8 to 8", "saturday": "9 to 5", "sunday": "closed",
}


@tool(timeout_seconds=8)
async def check_stock(item_said: str) -> dict:
    """Check whether the shop has something, and where it is.

    Args:
        item_said: What the caller asked for, in their own words.
    """
    words = {word for word in item_said.lower().split() if len(word) > 2}
    for name, row in STOCK.items():
        if words & set(name.split()):
            print(f"  [tool] check_stock({item_said!r}) -> {name}", flush=True)
            return {"found": True, "item": name, **row}
    print(f"  [tool] check_stock({item_said!r}) -> nothing", flush=True)
    return {"found": False, "stocked_items": sorted(STOCK)}


@tool(timeout_seconds=8)
async def opening_hours(day_said: str = "") -> dict:
    """Say when the shop is open, for one day or for the week.

    Args:
        day_said: The day the caller asked about, if they named one.
    """
    day = next((name for name in HOURS if name in (day_said or "").lower()), None)
    print(f"  [tool] opening_hours({day_said!r}) -> {day or 'the week'}", flush=True)
    return {"day": day, "hours": HOURS[day]} if day else {"week": HOURS}


@tool(timeout_seconds=8)
async def reserve_item(item: str, quantity_said: str, name: str) -> dict:
    """Hold something behind the counter for a caller to collect.

    Args:
        item: The item, exactly as check_stock returned it.
        quantity_said: How many they asked for, in their words.
        name: The caller's name.
    """
    quantity = int(digits_said(quantity_said) or 1)
    print(f"  [tool] reserve_item({item!r}, {quantity}, {name!r})", flush=True)
    row = STOCK.get(item)
    if row is None:
        return {"reserved": False, "reason": "we do not stock that"}
    if row["in_stock"] < quantity:
        return {"reserved": False, "reason": f"only {row['in_stock']} left"}
    row["in_stock"] -= quantity
    return {"reserved": True, "item": item, "quantity": quantity, "collect_by": "six tomorrow"}


TOOLS = [check_stock, opening_hours, reserve_item]

# With no reply endpoint, this prompt is the only thing shaping what is said,
# so it carries more weight than it would otherwise.
SYSTEM_PROMPT = """
You answer the phone for Ridgeway Hardware, a small tool shop.

Look things up rather than guessing: use check_stock for anything about
whether an item is in, opening_hours for when the shop is open, and
reserve_item only once the caller has said what they want, how many, and their
name. Never state a price, a quantity or an aisle that a tool did not return.

Keep every reply to one or two short sentences. Say numbers the way a person
would: two pounds ten, aisle four. If the shop does not stock something, say so
and offer what is nearby on the shelf.
"""


def build(base_url: str) -> VoiceAgent:
    """The declaration, with every tool pointed at this process."""

    # `hosted_at` returns a new tool bound to that address; the module-level
    # TOOLS are left unbound, so tests can import them and one run's tunnel
    # cannot leak into another declaration.
    auth = HttpToolHeaderInput(name="Authorization", value=f"Bearer {SECRET}")
    tools = [
        declared.hosted_at(f"{base_url}/tools/{declared.name}", headers=[auth])
        for declared in TOOLS
    ]

    return VoiceAgent(
        name="Ridgeway Hardware",
        voice=os.environ.get("VOICE", "ivy"),
        system_prompt=SYSTEM_PROMPT,
        greeting="Ridgeway Hardware, how can I help?",
        tools=tools,
        # No `llm=`: the platform's own model runs the conversation.
    )


def deploy(agent: VoiceAgent) -> str:
    client = Client(base_url=os.environ.get("AAI_BASE_URL", "https://agents.assemblyai.com"))
    stored = ID_FILE.read_text().strip() if ID_FILE.exists() else ""
    if stored:
        try:
            client.agents.update(stored, agent)
            print(f"updated agent {stored}")
            return stored
        except NotFoundError:
            pass
    created = client.agents.create(agent)
    ID_FILE.write_text(created.id)
    print(f"created agent {created.id}")
    return created.id


def main() -> int:
    if not os.environ.get("ASSEMBLYAI_API_KEY"):
        sys.exit("set ASSEMBLYAI_API_KEY")

    with public_address(PORT) as base_url:
        # Before the deploy, not after: deploy repoints the stored agent, and a
        # port already held by an earlier run would otherwise leave a live agent
        # whose tool URLs answer to nothing.
        claim_port(port=PORT)
        agent = build(base_url)
        agent_id = deploy(agent)
        print(f"\nagent {agent_id} is live. Point a phone number at it and call in.\n")
        # No `reply=`: only the tools are served from here.
        serve(agent, port=PORT, tool_secret=SECRET)
    return 0


if __name__ == "__main__":
    sys.exit(main())
