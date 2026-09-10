"""Your backend: serves the agent's tools, its pre-connect lookup and webhooks.

    pip install fastapi uvicorn
    export TOOL_SECRET=...          # same value pizza_line.py puts in the tool headers
    export WEBHOOK_SECRET=...       # same value you used to create the webhook subscription
    uvicorn server:app --port 8000  # run from the examples/ directory

Expose it over HTTPS (for example with a tunnel during development), set
``PUBLIC_BASE_URL`` to that address, and deploy the agent with deploy.py. The
platform then POSTs tool arguments to ``/tools/<name>`` when the model calls a
tool, POSTs to ``/pre-connect/whois`` before answering each phone call, and
POSTs webhook events to ``/webhooks/voice-agents``.
"""

import json
import logging
import os

from fastapi import FastAPI, HTTPException, Request, Response

from assemblyai_agents import WebhookVerificationError, verify
from pizza_line import agent

log = logging.getLogger("pizza_line")
logging.basicConfig(level=logging.INFO)

TOOL_SECRET = os.environ.get("TOOL_SECRET", "change-me")
WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET")

# The same Tool objects the declaration was built from: name -> handler.
TOOLS = {declared.name: declared for declared in agent.tools or []}

app = FastAPI(title="Pizza Line backend")


def _authorize(request: Request) -> None:
    # The header value configured on the tool (see pizza_line.hosted) arrives on
    # every call; refuse anything that does not carry it.
    if request.headers.get("Authorization") != f"Bearer {TOOL_SECRET}":
        raise HTTPException(status_code=401, detail="bad tool secret")


@app.post("/tools/{name}")
async def run_tool(name: str, request: Request):
    """Run one of the agent's tools with the arguments the model supplied.

    For POST tools the platform sends the arguments as the JSON body, shaped by
    the parameter schema derived from the function signature. Whatever JSON this
    returns is handed to the model as the tool result.
    """
    _authorize(request)
    declared = TOOLS.get(name)
    if declared is None:
        raise HTTPException(status_code=404, detail=f"no tool named {name!r}")
    arguments = await request.json()
    log.info("tool %s(%s)", name, arguments)
    try:
        return await declared.invoke(**arguments)
    except TypeError as exc:  # unexpected/missing argument
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/pre-connect/whois")
async def whois(request: Request):
    """Called before a phone call is answered; must finish within the timeout.

    Return the values named in the request's ``returns`` (here ``customer_tier``).
    Because the declaration sets ``allow_overrides=True``, a top-level
    ``greeting`` here replaces the agent's greeting for this call, and a
    top-level ``"reject": true`` would abort the call.
    """
    payload = await request.json() if int(request.headers.get("content-length") or 0) else {}
    log.info("pre-connect payload: %s", json.dumps(payload))
    return {
        "customer_tier": "gold",
        "greeting": "Pizza Palace, welcome back. How can I help?",
    }


@app.post("/webhooks/voice-agents")
async def webhook(request: Request):
    """Verify the signature over the raw body, then act on the event."""
    if WEBHOOK_SECRET is None:
        raise HTTPException(status_code=503, detail="WEBHOOK_SECRET not configured")
    body = await request.body()  # raw bytes, before any JSON parsing
    try:
        event = verify(body, request.headers.get("X-AAI-Signature", ""), WEBHOOK_SECRET)
    except WebhookVerificationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    log.info("webhook %s: %s", event.get("event") or event.get("type"), json.dumps(event)[:500])
    return Response(status_code=204)
