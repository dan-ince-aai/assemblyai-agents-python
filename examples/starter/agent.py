"""The agent: its tools, and the one declaration that deploys them.

Nothing here touches the network at import time, so the tests, the backend and
the deploy script can all import it.

Environment:
    PUBLIC_BASE_URL   public HTTPS address of backend.py (a tunnel in development)
    TOOL_SECRET       shared secret the platform presents on every tool call
    LLM_API_KEY       shared secret the platform presents on the reply endpoint
    AGENT_NAME        the name the agent gives (default "Sam")
    VOICE             a voice id (default "ivy")
"""

import os

from assemblyai_agents import Captured, Header, PreConnectRequest, VoiceAgent, tool
from assemblyai_agents.models.rest import (
    HttpMethod,
    HttpToolHeaderInput,
    LlmConfigRequest,
    PlaintextHttpToolConfig,
)

import store

PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "").rstrip("/")
TOOL_SECRET = os.environ.get("TOOL_SECRET", "change-me")
AGENT_NAME = os.environ.get("AGENT_NAME", "Sam")
VOICE = os.environ.get("VOICE", "ivy")


def hosted(path: str) -> PlaintextHttpToolConfig:
    """Where the platform fetches this tool from.

    Always over HTTPS: a phone call has no client on the line to ask, so the
    platform calls the tool itself. `run.py` works out the address first and
    then builds the declaration against it.
    """
    if not PUBLIC_BASE_URL:
        raise RuntimeError(
            "PUBLIC_BASE_URL is unset. Run `python run.py`, which gets an "
            "address before the declaration is built, or set it yourself."
        )
    return PlaintextHttpToolConfig(
        url=f"{PUBLIC_BASE_URL}{path}",
        http_method=HttpMethod.POST,
        headers=[HttpToolHeaderInput(name="Authorization", value=f"Bearer {TOOL_SECRET}")],
    )


# --------------------------------------------------------------------------- tools
#
# Two habits worth keeping, both learned from the platform refusing things:
#
# 1. Take the caller's words, not a value you derived from them. The platform
#    checks every argument against the conversation and refuses a call carrying
#    anything nobody said, so the reading belongs at this end.
# 2. Return what you read back. Whatever writes the reply sees the tool result
#    and nothing else, so a result that omits the id cannot name it out loud.


@tool(timeout_seconds=8, http=hosted("/tools/find_patient"))
async def find_patient(reference_said: str) -> dict:
    """Find the patient record from the reference number on their reminder.

    Returns nothing identifying: the caller states their own name and
    verify_caller checks it.

    Args:
        reference_said: The reference as the caller read it out, in their words.
    """
    patient = store.find_by_reference(reference_said)
    if patient is None:
        return {"found": False}
    return {"found": True, "reference": patient["reference"], "last_seen": patient["last_seen"]}


@tool(timeout_seconds=8, http=hosted("/tools/verify_caller"))
async def verify_caller(caller_said: str, reference: str = "") -> dict:
    """Check the name the caller gave against the name on the record.

    Pass exactly what the caller said, however they said it. Omit reference
    when they have not read one out: the record matched from the number they
    called from is used instead.

    Args:
        caller_said: Exactly what the caller said, word for word.
        reference: The patient reference, if the call has established one.
    """
    patient = store.resolve(reference)
    if patient is None:
        return {"verified": False, "problem": "no_record_in_context"}
    heard = store.extract_name(caller_said)
    if not heard:
        # They asked something rather than answering. No attempt is spent and
        # nothing is implied about their name.
        return {"verified": False, "problem": "no_name_heard"}
    if not store.name_matches(patient, heard):
        return {"verified": False, "problem": "no_match", "name_heard": heard}
    return {
        "verified": True,
        "reference": patient["reference"],
        "first_name": patient["first_name"],
        "balance_due": patient["balance_due"],
    }


@tool(timeout_seconds=8, http=hosted("/tools/find_appointments"))
async def find_appointments(preference_said: str = "") -> dict:
    """List the next free appointments, soonest first.

    Args:
        preference_said: What the caller said about when they would like to come in, if anything.
    """
    slots = store.open_slots(limit=4)
    return {
        "slots": [
            {**slot, "spoken": f"{store.spoken_date(slot['date'])} at {store.spoken_time(slot['time'])}"}
            for slot in slots
        ],
        "preference_noted": preference_said or None,
    }


@tool(timeout_seconds=10, http=hosted("/tools/book_appointment"))
async def book_appointment(slot_date: str, slot_time: str, reference: str = "") -> dict:
    """Book one of the free appointments for this patient.

    Use a date and time exactly as find_appointments returned them.

    Args:
        slot_date: The appointment date, as YYYY-MM-DD.
        slot_time: The appointment time, as HH:MM.
        reference: The patient reference, if the call has established one.
    """
    patient = store.resolve(reference)
    if patient is None:
        return {"booked": False, "problem": "no_record_in_context"}
    result = store.book(patient, slot_date, slot_time)
    if result.get("booked"):
        result["spoken"] = f"{store.spoken_date(slot_date)} at {store.spoken_time(slot_time)}"
    return result


@tool(timeout_seconds=10, http=hosted("/tools/request_callback"))
async def request_callback(reason: str, note: str = "", reference: str = "") -> dict:
    """Ask a member of the practice team to call this patient back.

    Args:
        reason: Why a person needs to call, in a few words.
        note: Anything the caller said that the team should see.
        reference: The patient reference, if the call has established one.
    """
    return store.request_callback(store.resolve(reference), reason, note)


TOOLS = [find_patient, verify_caller, find_appointments, book_appointment, request_callback]


# --------------------------------------------------------------------------- the declaration
#
# The prompt is short on purpose. With `llm=` set, reply.py decides every word,
# so the prompt is only what the platform prepends to what your endpoint sees.

SYSTEM_PROMPT = f"""
You are {AGENT_NAME}, on the phone for {store.PRACTICE}, on a recorded line.
Keep every reply to one or two short sentences and ask one question at a time.
Confirm who you are speaking to before discussing anything on their record.
"""


def byo_llm() -> LlmConfigRequest | None:
    """Point reply generation at this backend, so reply.py owns the words."""
    base_url = os.environ.get("LLM_BASE_URL", "").rstrip("/")
    if not base_url and os.environ.get("BYO_LLM") and PUBLIC_BASE_URL:
        base_url = f"{PUBLIC_BASE_URL}/v1"
    if not base_url:
        return None
    return LlmConfigRequest(
        base_url=base_url,
        model=os.environ.get("LLM_MODEL", "starter-reply-engine"),
        api_key=os.environ.get("LLM_API_KEY", "change-me"),
    )


def pre_connect() -> list[PreConnectRequest] | None:
    """Look the caller up before a phone call is answered.

    Telephony only: a WebSocket session never runs this and the flow falls back
    to asking for the reference. It fails open, so a slow lookup costs the
    personalised greeting and nothing else. `allow_overrides` is what lets the
    response replace the greeting.
    """
    if not PUBLIC_BASE_URL:
        return None
    return [
        PreConnectRequest(
            url=f"{PUBLIC_BASE_URL}/pre-connect/lookup",
            headers=[Header(name="Authorization", value=f"Bearer {TOOL_SECRET}")],
            returns=[
                Captured(name="reference", path="reference", default=""),
                Captured(name="first_name", path="first_name", default=""),
            ],
            timeout_ms=800,
            allow_overrides=True,
        )
    ]


agent = VoiceAgent(
    name=f"{store.PRACTICE} reception",
    voice=VOICE,
    system_prompt=SYSTEM_PROMPT,
    greeting=(
        f"Thank you for calling {store.PRACTICE} on a recorded line. "
        f"My name is {AGENT_NAME}. How can I help?"
    ),
    tools=TOOLS,
    llm=byo_llm(),
    pre_connect=pre_connect(),
)
