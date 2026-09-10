"""The service the platform calls. There is no web framework here.

    export TOOL_SECRET=... LLM_API_KEY=...
    python backend.py

`serve()` answers the platform's requests with the functions in this project:
each `@tool` on the declaration, the `decide` in reply.py, and the pre-connect
handler below. Nothing to install beyond the SDK, and no service to write.

    POST /tools/{name}            the agent's tools
    POST /v1/chat/completions     reply.decide, streamed
    POST /pre-connect/lookup      before a phone call is answered
    POST /webhooks/voice-agents   verified against WEBHOOK_SECRET
    GET  /healthz

If you would rather host these in your own application, `routes()` from the
same module hands back the identical handlers as plain callables.
"""

import os

from assemblyai_agents.serving import serve

import reply
import store
from agent import AGENT_NAME, agent

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

    Says nothing from their record: nobody has confirmed who picked up yet.
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


if __name__ == "__main__":
    serve(
        agent,
        reply=reply.decide,
        port=int(os.environ.get("PORT", "8000")),
        tool_secret=TOOL_SECRET,
        llm_key=LLM_API_KEY,
        pre_connect={"/pre-connect/lookup": lookup},
        webhook_secret=os.environ.get("WEBHOOK_SECRET"),
        on_event=lambda event: print(f"[webhook] {event.get('event') or event.get('type')}"),
    )
