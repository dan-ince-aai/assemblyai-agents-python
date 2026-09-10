"""Run a whole call locally: no platform, no network, no microphone.

The loop here is the platform's loop. Ask the reply engine what to say, run any
tool it asks for, hand the result back, ask again. It is the fastest way to see
a change, and the tests drive the same function.

    python rehearse.py                  the happy path
    python rehearse.py objection        an objection the script does not cover
    python rehearse.py wrong_name       verification that fails
"""

import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
# The declaration reads these when it is imported, and a rehearsal never fetches
# anything, so a placeholder address is the honest value here. `run.py` sets the
# real one before importing, and that wins.
os.environ.setdefault("PUBLIC_BASE_URL", "https://rehearsal.invalid")
os.environ.setdefault("TOOL_SECRET", "rehearsal-tool-secret")
os.environ.setdefault("LLM_API_KEY", "rehearsal-llm-key")
os.environ.setdefault("BYO_LLM", "1")

from assemblyai_agents.byo import Call, Say, Turn

import reply
import store
from agent import TOOLS, agent

BY_NAME = {declared.name: declared for declared in TOOLS}

# Roughly what the platform appends to the system prompt on every turn.
GUIDANCE = "\n\nOutput is spoken aloud. Plain conversational text only. Spell identifiers digit by digit.\n"


def run(
    caller_lines: list[str],
    caller_number: str | None = None,
    quiet: bool = False,
    prepare=None,
) -> list[tuple[str, str]]:
    """Play a call. `prepare` runs after the reset, to set the world up."""
    store.reset()
    reply.memo.forget()
    if prepare is not None:
        prepare()
    transcript: list[tuple[str, str]] = []
    greeting = agent.greeting
    messages: list[dict] = [{"role": "system", "content": agent.system_prompt + GUIDANCE}]

    if caller_number:
        patient = store.find_by_phone(caller_number)
        store.remember_in_flight(patient)
        if patient is not None:
            # The platform runs the pre-connect lookup itself and puts what it
            # captured into the transcript as this tool's result.
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "pc",
                    "type": "function",
                    "function": {"name": "aai_pre_connect_context", "arguments": "{}"},
                }],
            })
            messages.append({
                "role": "tool",
                "tool_call_id": "pc",
                "content": json.dumps({"variables": {
                    "reference": patient["reference"],
                    "first_name": patient["first_name"],
                }}),
            })
            greeting = (
                f"Thank you for calling {store.PRACTICE} on a recorded line. I have found your "
                f"record from the number you are calling from. Could you give me your full name?"
            )

    messages.append({"role": "assistant", "content": greeting})
    transcript.append(("agent", greeting))
    tools = [{"type": "function", "function": {"type": "function", "name": name, "parameters": {}}} for name in BY_NAME]

    for line in caller_lines:
        messages.append({"role": "user", "content": line})
        transcript.append(("caller", line))
        for _ in range(6):
            turn = Turn.from_request({"model": "rehearsal", "messages": messages, "tools": tools})
            action = reply.decide(turn)
            if isinstance(action, Call):
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": f"c{len(messages)}",
                        "type": "function",
                        "function": {"name": action.name, "arguments": json.dumps(action.arguments)},
                    }],
                })
                result = asyncio.run(BY_NAME[action.name].invoke(**action.arguments))
                messages.append({
                    "role": "tool",
                    "tool_call_id": f"c{len(messages) - 1}",
                    "content": json.dumps(result),
                })
                transcript.append(("tool", f"{action.name}({json.dumps(action.arguments)}) -> {json.dumps(result)}"))
                # The platform adds its own note once a tool completes.
                messages.append({"role": "system", "content": f"The function call {action.name} has just completed."})
                continue
            text = action.text if isinstance(action, Say) else ""
            messages.append({"role": "assistant", "content": text})
            transcript.append(("agent", text or "(silence)"))
            break

    if not quiet:
        for speaker, text in transcript:
            print(f"    · {text}" if speaker == "tool" else f"{'AGENT ' if speaker == 'agent' else 'CALLER'}: {text}")
    return transcript


HAPPY = [
    "Hi, I need to book a check-up.",
    "It's four four seven one.",
    "Yes, this is Maria Delgado.",
    "Some time in the morning would be good.",
    "The first one works.",
]

BY_PHONE = ["Yes, Maria Delgado speaking.", "Tomorrow morning if you have it.", "Yes please."]

WRONG_NAME = [
    "It's four four seven one.",
    "This is John Smith.",
    "What name have you got?",
    "Fine, John Smith again.",
    "Still John Smith.",
]

OBJECTION = [
    "It's four four seven one.",
    "Maria Delgado.",
    "Some time this week.",
    "Nothing sooner than that? My tooth is killing me.",
]

SCENARIOS = {
    "happy": (HAPPY, None),
    "by_phone": (BY_PHONE, "+14695550142"),
    "wrong_name": (WRONG_NAME, None),
    "objection": (OBJECTION, None),
}

if __name__ == "__main__":
    for index, name in enumerate(sys.argv[1:] or ["happy"]):
        if name not in SCENARIOS:
            sys.exit(f"unknown scenario {name!r}; pick from {', '.join(SCENARIOS)}")
        lines, number = SCENARIOS[name]
        print(("\n" if index else "") + f"=== {name} " + "=" * (58 - len(name)))
        run(lines, caller_number=number)
        if store.APPOINTMENTS:
            print(f"    appointments: {store.APPOINTMENTS}")
        if store.CALLBACKS:
            print(f"    callbacks: {store.CALLBACKS}")
