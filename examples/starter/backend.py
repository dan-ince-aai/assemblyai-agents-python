"""The service the platform calls. Four routes, written out.

    pip install fastapi uvicorn
    export TOOL_SECRET=... LLM_API_KEY=...
    uvicorn backend:app --port 8000

    POST /v1/chat/completions   asked what to say on every turn -> reply.py
    POST /tools/{name}          runs one of the agent's tools
    POST /pre-connect/lookup    before a phone call is answered; may rewrite the greeting
    POST /webhooks/voice-agents signed session and call events
    GET  /healthz               what is loaded

Nothing here is clever, on purpose: this is the file you will want to read when
something does not arrive, so it says what it does.
"""

import json
import os

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from assemblyai_agents import WebhookVerificationError, verify
from assemblyai_agents.byo import Turn, json_body, stream

import model
import reply
import store
from agent import AGENT_NAME, TOOLS, agent

TOOL_SECRET = os.environ.get("TOOL_SECRET", "change-me")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "change-me")
WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET")
# Treat any caller as the demo patient, so a call from any handset reaches the
# personalised greeting. Off by default.
DEMO_MATCH_ANY = os.environ.get("DEMO_MATCH_ANY", "") not in ("", "0", "false")

BY_NAME = {declared.name: declared for declared in TOOLS}
app = FastAPI(title=f"{store.PRACTICE} voice backend")


def authorize(request: Request, secret: str) -> None:
    if request.headers.get("Authorization") != f"Bearer {secret}":
        raise HTTPException(status_code=401, detail="unauthorized")


@app.get("/healthz")
def healthz():
    return {
        "ok": True,
        "agent": agent.name,
        "tools": sorted(BY_NAME),
        "replies_from_here": bool(agent.llm),
        "stages": list(reply.stages.names),
        "model": model.status(),
    }


@app.post("/v1/chat/completions")
async def replies(request: Request):
    """What to say next. Every request arrives streaming, so the answer does too."""
    if request.headers.get("Authorization") != f"Bearer {LLM_API_KEY}":
        return JSONResponse({"error": {"message": "bad api key"}}, status_code=401)
    body = await request.json()
    turn = Turn.from_request(body)
    answer = reply.decide(turn)
    stage = reply.stages.current(turn)
    print(f"[reply] stage={stage.name if stage else '-'} said={turn.caller_said[:48]!r} -> {answer}", flush=True)
    if body.get("stream"):
        return StreamingResponse(stream(turn, answer), media_type="text/event-stream")
    return JSONResponse(json_body(turn, answer))


@app.post("/tools/{name}")
async def run_tool(name: str, request: Request):
    """Run a tool with the arguments the reply engine asked for."""
    authorize(request, TOOL_SECRET)
    declared = BY_NAME.get(name)
    if declared is None:
        raise HTTPException(status_code=404, detail=f"no tool named {name!r}")
    arguments = await request.json()
    print(f"[tool] {name}({json.dumps(arguments)[:120]})", flush=True)
    try:
        result = await declared.invoke(**arguments)
    except TypeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    print(f"[tool] {name} -> {json.dumps(result, default=str)[:160]}", flush=True)
    return result


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


@app.post("/pre-connect/lookup")
async def lookup(request: Request):
    """Find the caller before the call is answered, and greet them by name.

    Says nothing from their record: nobody has confirmed who picked up yet.
    """
    authorize(request, TOOL_SECRET)
    raw = await request.body()
    payload = json.loads(raw) if raw else {}
    number = caller_number(payload) or os.environ.get("DEMO_CALLER_NUMBER", "")
    patient = store.find_by_phone(number) if number else None
    if patient is None and DEMO_MATCH_ANY:
        patient = next(iter(store.PATIENTS.values()))
    store.remember_in_flight(patient)
    reply.memo.forget()  # a new call starts with nothing remembered
    print(f"[pre-connect] number={number!r} matched={'yes' if patient else 'no'}", flush=True)
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


@app.post("/webhooks/voice-agents")
async def webhook(request: Request):
    """Session and call events, signed. Verify over the raw body before parsing."""
    if WEBHOOK_SECRET is None:
        raise HTTPException(status_code=503, detail="WEBHOOK_SECRET not configured")
    body = await request.body()
    try:
        event = verify(body, request.headers.get("X-AAI-Signature", ""), WEBHOOK_SECRET)
    except WebhookVerificationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    print(f"[webhook] {event.get('event') or event.get('type')}", flush=True)
    return Response(status_code=204)
