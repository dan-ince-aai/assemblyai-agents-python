"""The service the platform calls, mounted in one call.

    pip install fastapi uvicorn
    export TOOL_SECRET=... LLM_API_KEY=...
    uvicorn backend:app --port 8000

`mount_fastapi` adds every surface the platform reaches for:

    POST /v1/chat/completions   asked what to say on every turn -> reply.py
    POST /tools/{name}          runs one of the agent's tools
    POST /pre-connect/lookup    before a phone call is answered, may rewrite the greeting
    POST /webhooks/voice-agents signed session and call events
    GET  /healthz               what is loaded

Everything domain-specific is in reply.py, agent.py and store.py. This file is
wiring, and it stays this short as the agent grows.
"""

import os

from fastapi import FastAPI

from assemblyai_agents.byo import mount_fastapi

import model
import reply
import store
from agent import AGENT_NAME, TOOLS, agent

TOOL_SECRET = os.environ.get("TOOL_SECRET", "change-me")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "change-me")
# Treat any caller as the demo patient, so a call from any handset reaches the
# personalised greeting. Off by default.
DEMO_MATCH_ANY = os.environ.get("DEMO_MATCH_ANY", "") not in ("", "0", "false")

# The platform's pre-connect request arrives with an empty body today, so there
# is no caller number in it to look up. Every plausible field is checked anyway,
# and the lookup fails open, which is why the greeting still works without one.
CALLER_KEYS = ("from", "from_number", "caller", "caller_id", "ani", "phone_number")


def caller_number(payload: dict) -> str | None:
    for key in CALLER_KEYS:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value
    for value in payload.values():
        if isinstance(value, dict):
            found = caller_number(value)
            if found:
                return found
    return None


def lookup(payload: dict) -> dict:
    """Find the caller before the call is answered, and greet them by name.

    Says nothing from the record: nobody has confirmed who picked up yet.
    """
    number = caller_number(payload) or os.environ.get("DEMO_CALLER_NUMBER", "")
    patient = store.find_by_phone(number) if number else None
    if patient is None and DEMO_MATCH_ANY:
        patient = next(iter(store.PATIENTS.values()))
    store.remember_in_flight(patient)
    reply.memo.forget()  # a new call starts with nothing remembered
    if patient is None:
        return {"matched": False}
    return {
        "matched": True,
        "reference": patient["reference"],
        "first_name": patient["first_name"],
        "greeting": (
            f"Thank you for calling {store.PRACTICE} on a recorded line. My name is "
            f"{AGENT_NAME}. I have found your record from the number you are calling "
            f"from. Could you give me your full name so I can check it?"
        ),
    }


app = FastAPI(title=f"{store.PRACTICE} voice backend")

mount_fastapi(
    app,
    reply.responder,
    tools=TOOLS,
    tool_secret=TOOL_SECRET,
    llm_key=LLM_API_KEY,
    pre_connect={"/pre-connect/lookup": lookup},
    webhook_secret=os.environ.get("WEBHOOK_SECRET"),
)


@app.get("/")
def about():
    """What this backend is serving, for a quick look after deploying."""
    return {
        "agent": agent.name,
        "replies_from_here": bool(agent.llm),
        "stages": list(reply.responder.names),
        "model": model.status(),
    }
