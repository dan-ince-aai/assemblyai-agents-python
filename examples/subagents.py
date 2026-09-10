"""Subagent routing: a different model, prompt and tool set for each stage.

A call is not one job. Checking who is on the line is a narrow, scripted task
that a small model does perfectly and cheaply. Talking someone through a
booking is not. Handling a complaint is different again. Giving all of it to
one model and one prompt means paying for the hardest turn on every turn, and
hoping a single instruction sheet covers every case.

So this splits the call into subagents. Each one owns a stage and carries:

    model    a gateway model chosen for that stage's difficulty
    prompt   instructions for that stage only
    tools    the only tools it is allowed to call
    done     when it hands over

    authenticate   claude-haiku-4-5   verify_caller                  cheap, narrow
    booking        claude-sonnet-4-6  find_appointments, book_...    the real conversation
    recovery       claude-opus-5      find_appointments, escalate    only when it goes wrong
    wrap_up        claude-haiku-4-5   (none)                         say goodbye

Two things are worth being precise about.

**Switching subagent is not a config change.** With `llm=` pointing here, this
process decides every reply, so a handover is this code choosing a different
model and prompt for the next turn. Nothing is redeployed and the platform is
not told. (`session.update` can change a live session's config, but it is a
WebSocket message and a phone call has no client to send it, so it is not the
mechanism here.)

**The tool allowlist is enforced here, not by the platform.** The platform
executes whatever tool call it is handed. So `authenticate` cannot book an
appointment because this file refuses to pass such a call on, and logs it. That
is a real guarantee, but it is yours to keep.

    export ASSEMBLYAI_API_KEY=...
    python subagents.py
"""

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import httpx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from assemblyai_agents import Client, NotFoundError, VoiceAgent, tool
from assemblyai_agents.byo import Turn, call_tool, digits_said, say, silence
from assemblyai_agents.models.rest import (
    HttpMethod,
    HttpToolHeaderInput,
    LlmConfigRequest,
    PlaintextHttpToolConfig,
)
from assemblyai_agents.serving import serve

from expose import public_address

PORT = int(os.environ.get("PORT", "8000"))
SECRET = os.environ.get("SHARED_SECRET", "change-me-" + os.urandom(4).hex())
ID_FILE = Path(__file__).with_name(".subagents_id")

# Any OpenAI-compatible gateway. This one takes the same key as the rest of the
# platform, which is why there is no second provider to set up.
GATEWAY = os.environ.get("MODEL_BASE_URL", "https://llm-gateway.assemblyai.com/v1")
GATEWAY_KEY = os.environ.get("MODEL_API_KEY") or os.environ.get("ASSEMBLYAI_API_KEY", "")
# The platform gives a reply endpoint about ten seconds, and the caller hears
# silence while it waits.
MODEL_TIMEOUT = float(os.environ.get("MODEL_TIMEOUT", "6"))

PATIENTS = {"4471": "Maria Delgado", "8820": "Tomas Okonkwo"}
SLOTS = ["Friday at nine", "Friday at half past eleven", "Monday at two"]
BOOKINGS: list[dict] = []
ESCALATIONS: list[dict] = []


# --------------------------------------------------------------------------- tools


@tool(timeout_seconds=8)
async def verify_caller(reference_said: str, name_said: str) -> dict:
    """Check a caller's reference and name against the record.

    Args:
        reference_said: The reference as the caller said it.
        name_said: The name as the caller said it.
    """
    reference = digits_said(reference_said)[-4:]
    expected = PATIENTS.get(reference)
    ok = bool(expected) and expected.split()[-1].lower() in name_said.lower()
    print(f"  [tool] verify_caller({reference!r}, {name_said!r}) -> {ok}", flush=True)
    return {"verified": ok, "reference": reference if ok else None}


@tool(timeout_seconds=8)
async def find_appointments() -> dict:
    """List the appointments that are still free."""
    free = [slot for slot in SLOTS if slot not in {b["slot"] for b in BOOKINGS}]
    print(f"  [tool] find_appointments() -> {len(free)} free", flush=True)
    return {"slots": free}


@tool(timeout_seconds=10)
async def book_appointment(slot: str, reference: str) -> dict:
    """Book one of the free appointments.

    Args:
        slot: The appointment, worded exactly as find_appointments returned it.
        reference: The patient reference, from verify_caller.
    """
    if slot not in SLOTS or slot in {b["slot"] for b in BOOKINGS}:
        return {"booked": False, "reason": "that one has gone"}
    BOOKINGS.append({"slot": slot, "reference": reference})
    print(f"  [tool] book_appointment({slot!r}) -> booked", flush=True)
    return {"booked": True, "slot": slot, "confirmation": f"R{len(BOOKINGS):03d}"}


@tool(timeout_seconds=8)
async def escalate(reason: str) -> dict:
    """Hand the caller to a person, with a note of why.

    Args:
        reason: Why this needs a human, in a sentence.
    """
    ESCALATIONS.append({"reason": reason})
    print(f"  [tool] escalate({reason!r})", flush=True)
    return {"escalated": True}


TOOLS = [verify_caller, find_appointments, book_appointment, escalate]


# --------------------------------------------------------------------------- the subagents


# The platform appends its own spoken-output guidance to an agent's system
# prompt, and a subagent prompt replaces the whole thing, so that guidance is
# lost unless it is put back. Without this a model writes for a screen: the
# booking subagent offered "**Friday at nine**", asterisks and all, which text
# to speech reads out or mangles.
VOICE_RULES = (
    "You are speaking on a phone call and everything you say is read aloud. "
    "Plain words only: no asterisks, bullets, markdown or emoji. Say numbers as a "
    "person would. Keep to one or two short sentences and ask one thing at a time."
)


@dataclass(frozen=True)
class Subagent:
    name: str
    model: str
    prompt: str
    tools: tuple
    done: Callable[[Turn], bool]

    @property
    def instructions(self) -> str:
        return f"{VOICE_RULES}\n\n{self.prompt}"


def verified(turn: Turn) -> bool:
    result = turn.result_of("verify_caller")
    return bool(result and result.get("verified"))


def booked(turn: Turn) -> bool:
    result = turn.result_of("book_appointment")
    return bool(result and result.get("booked"))


def struggling(turn: Turn) -> bool:
    """Three tool results with nothing booked: this is not going well."""
    return len(turn.results) >= 4 and not booked(turn)


SUBAGENTS = [
    Subagent(
        name="authenticate",
        # A narrow, scripted job. The cheapest model does it as well as any.
        model=os.environ.get("MODEL_AUTH", "claude-haiku-4-5-20251001"),
        prompt=(
            "You are checking who is on the line for Fairview Dental, and nothing else.\n"
            "Ask for the four digit reference from their reminder, then their full name. "
            "They will give them in separate turns, so read back through the whole "
            "conversation for both before calling anything.\n"
            "Once you have both, call verify_caller, passing the caller's exact words: "
            "the turn where they gave the reference as reference_said, the turn where "
            "they gave their name as name_said. Never call it with an empty value; ask "
            "again for whichever one is missing.\n"
            "Say nothing about appointments or records until it comes back verified. "
            "One short sentence per reply."
        ),
        tools=("verify_caller",),
        done=verified,
    ),
    Subagent(
        name="booking",
        # The part that is an actual conversation.
        model=os.environ.get("MODEL_BOOKING", "claude-sonnet-4-6"),
        prompt=(
            "You are booking a dental appointment for a caller whose identity is already "
            "confirmed. Call find_appointments to see what is free, offer at most two, and "
            "call book_appointment once they choose, using the reference from the earlier "
            "verify_caller result. Read the confirmation back. One or two short sentences."
        ),
        tools=("find_appointments", "book_appointment"),
        done=lambda turn: booked(turn) or struggling(turn),
    ),
    Subagent(
        name="recovery",
        # Rare, and worth the strongest model when it happens.
        model=os.environ.get("MODEL_RECOVERY", "claude-opus-5"),
        prompt=(
            "A booking has gone wrong: slots were offered and nothing was booked. Work out "
            "what the caller actually needs. Offer what is genuinely free, or call escalate "
            "with a clear reason if a person is required. Do not promise anything a tool has "
            "not returned."
        ),
        tools=("find_appointments", "escalate"),
        done=lambda turn: booked(turn) or bool(turn.result_of("escalate")),
    ),
    Subagent(
        name="wrap_up",
        model=os.environ.get("MODEL_WRAP", "claude-haiku-4-5-20251001"),
        prompt="The call is finished. Say one short goodbye and nothing else.",
        tools=(),
        done=lambda turn: False,
    ),
]


def route(turn: Turn) -> Subagent:
    """Whose turn it is. The first subagent that has not finished its job."""
    for subagent in SUBAGENTS:
        if not subagent.done(turn):
            return subagent
    return SUBAGENTS[-1]


# --------------------------------------------------------------------------- asking a model


def schemas_for(turn: Turn, allowed: tuple) -> list:
    """The tool schemas this subagent may see, in plain OpenAI shape.

    The platform nests a second `type` and its own fields inside `function`, so
    the schema is rebuilt rather than forwarded.
    """
    offered = []
    for entry in turn.request.get("tools") or []:
        function = entry.get("function") or {}
        if function.get("name") not in allowed:
            continue
        offered.append({
            "type": "function",
            "function": {
                "name": function.get("name"),
                "description": function.get("description") or "",
                "parameters": function.get("parameters") or {"type": "object", "properties": {}},
            },
        })
    return offered


def conversation(turn: Turn, subagent: Subagent) -> list:
    """The call so far, in a form another model will accept.

    Tool exchanges are flattened into notes rather than forwarded as
    function-call blocks. That is the part worth understanding: each subagent
    is offered a different set of tools, and a model refuses a transcript
    containing calls to tools it has not been given. Prior calls are history,
    not instructions, so they read better as narrative anyway.

    Two smaller fixes, both learned from a gateway refusing the request: the
    platform's spoken lines come back with a trailing space and an assistant
    message may not end in whitespace, and the list has to end on a user turn.
    """
    messages = [{"role": "system", "content": subagent.instructions}]
    calls: dict = {}

    def note(text: str) -> None:
        messages.append({"role": "user", "content": text})

    for message in turn.messages[1:]:
        role = message.get("role")
        content = message.get("content")
        if isinstance(content, str):
            content = content.strip()

        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            calls[call.get("id")] = function.get("name")
            note(f"(you called {function.get('name')} with {function.get('arguments') or '{}'})")

        if role == "tool":
            name = calls.get(message.get("tool_call_id"), "a tool")
            note(f"({name} returned {content})")
        elif role == "system" and content:
            note(f"(note: {content})")
        elif role == "assistant" and content:
            messages.append({"role": "assistant", "content": content})
        elif role == "user" and content:
            messages.append({"role": "user", "content": content})

    if messages[-1].get("role") != "user":
        note(turn.caller_said or "(the caller is waiting)")
    return messages


def ask(subagent: Subagent, turn: Turn):
    """One completion from this subagent's model. Returns text or a tool call."""
    body = {
        "model": subagent.model,
        "max_tokens": 300,
        "messages": conversation(turn, subagent),
    }
    schemas = schemas_for(turn, subagent.tools)
    if schemas:
        body["tools"] = schemas
    response = httpx.post(
        f"{GATEWAY}/chat/completions",
        headers={"Authorization": f"Bearer {GATEWAY_KEY}", "Content-Type": "application/json"},
        json=body,
        timeout=MODEL_TIMEOUT,
    )
    if response.status_code != 200:
        print(f"  [{subagent.name}] gateway {response.status_code}: {response.text[:120]}", flush=True)
        return None
    message = response.json()["choices"][0]["message"]
    calls = message.get("tool_calls") or []
    if calls:
        function = calls[0]["function"]
        name = function["name"]
        try:
            arguments = json.loads(function.get("arguments") or "{}")
        except ValueError:
            arguments = {}
        # The platform runs whatever it is handed, so the allowlist is kept
        # here or it is not kept at all.
        if name not in subagent.tools:
            print(f"  [{subagent.name}] refused out-of-scope tool {name}", flush=True)
            return say("Let me check that for you.")
        return call_tool(name, **arguments)
    return say((message.get("content") or "").strip() or "Sorry, could you say that again?")


def decide(turn: Turn):
    subagent = route(turn)
    answer = ask(subagent, turn)
    if answer is None:
        # A model that is slow or down must not take the call with it.
        return say("Sorry, I am having trouble here. Let me get someone to help.")
    detail = getattr(answer, "name", None) or f"{getattr(answer, 'text', '')[:60]!r}"
    print(f"[{subagent.name} · {subagent.model}] -> {detail}", flush=True)
    return answer


# --------------------------------------------------------------------------- wiring


def build(base_url: str) -> VoiceAgent:
    def hosted(path: str) -> PlaintextHttpToolConfig:
        return PlaintextHttpToolConfig(
            url=f"{base_url}{path}",
            http_method=HttpMethod.POST,
            headers=[HttpToolHeaderInput(name="Authorization", value=f"Bearer {SECRET}")],
        )

    for declared in TOOLS:
        object.__setattr__(declared.spec, "http", hosted(f"/tools/{declared.name}"))

    return VoiceAgent(
        name="Fairview Dental (subagents)",
        voice=os.environ.get("VOICE", "ivy"),
        # Every tool is declared on the agent. Which of them a given turn may
        # use is decided here, not there.
        system_prompt="You answer the phone for Fairview Dental.",
        greeting="Fairview Dental, how can I help?",
        tools=TOOLS,
        llm=LlmConfigRequest(base_url=f"{base_url}/v1", model="subagent-router", api_key=SECRET),
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
    print("subagents:")
    for subagent in SUBAGENTS:
        print(f"  {subagent.name:13} {subagent.model:28} tools: {', '.join(subagent.tools) or '(none)'}")
    with public_address(PORT) as base_url:
        agent = build(base_url)
        agent_id = deploy(agent)
        print(f"\nagent {agent_id} is live. Point a phone number at it and call in.\n")
        serve(agent, reply=decide, port=PORT, tool_secret=SECRET, llm_key=SECRET)
    return 0


if __name__ == "__main__":
    sys.exit(main())
