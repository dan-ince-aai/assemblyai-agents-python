"""The agent declaration shared by the other examples.

Nothing in this module talks to the network, so it is safe to import from tests.

Tools are declared once and can run in two places:

* With ``PUBLIC_BASE_URL`` set (the public HTTPS address of ``server.py``), each
  tool carries an ``http=`` config and the platform calls your backend when the
  model invokes it. This is the production shape and the only one that works on
  phone calls. A pre-connect request is declared the same way.
* With ``PUBLIC_BASE_URL`` unset, the tools are *client-resident*: the model's
  call is delivered over the WebSocket to whichever process is connected
  (``talk.py``) and the function runs there. Handy for local development.
"""

import os

from assemblyai_agents import Captured, PreConnectRequest, VoiceAgent, tool
from assemblyai_agents.models.rest import (
    HttpMethod,
    HttpToolHeaderInput,
    PlaintextHttpToolConfig,
)

PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "").rstrip("/")
# Shared secret the platform presents to server.py on every tool call.
TOOL_SECRET = os.environ.get("TOOL_SECRET", "change-me")

# Stand-in for your order system.
ORDERS = {
    "W004": {"status": "shipped", "eta": "Thursday"},
    "W005": {"status": "preparing", "eta": "20 minutes"},
}


def hosted(path: str) -> PlaintextHttpToolConfig | None:
    """Point the platform at server.py, or return None to run the tool in-process."""
    if not PUBLIC_BASE_URL:
        return None
    return PlaintextHttpToolConfig(
        url=f"{PUBLIC_BASE_URL}{path}",
        http_method=HttpMethod.POST,
        headers=[HttpToolHeaderInput(name="Authorization", value=f"Bearer {TOOL_SECRET}")],
    )


@tool(timeout_seconds=10, http=hosted("/tools/lookup_order"))
async def lookup_order(order_id: str) -> dict:
    """Look up the status of a customer's order by its order number.

    Args:
        order_id: The order number the caller read out, like W004.
    """
    return ORDERS.get(order_id.upper(), {"error": f"no order {order_id}"})


@tool(timeout_seconds=10, http=hosted("/tools/cancel_order"))
async def cancel_order(order_id: str, reason: str = "customer request") -> dict:
    """Cancel an order that has not shipped yet.

    Args:
        order_id: The order number, like W005.
        reason: Why the caller wants to cancel, in their words.
    """
    order = ORDERS.get(order_id.upper())
    if order is None:
        return {"error": f"no order {order_id}"}
    if order["status"] == "shipped":
        return {"cancelled": False, "reason": "already shipped"}
    order["status"] = "cancelled"
    return {"cancelled": True, "order_id": order_id.upper(), "reason": reason}


agent = VoiceAgent(
    name="Pizza Line",
    voice="ivy",
    system_prompt="""
        You answer order questions for Pizza Palace.
        Keep replies to one or two short sentences.
        For any question about an order, call lookup_order and read back the
        status and ETA. Only cancel an order when the caller clearly asks to,
        and confirm the order number first. If an order is not found, ask the
        caller to repeat the number.
    """,
    greeting="Pizza Palace, how can I help?",
    tools=[lookup_order, cancel_order],
    # Looks the caller up before a phone call is answered (telephony only; a
    # WebSocket session skips it). Fails open: if the endpoint is slow or down,
    # the call proceeds without the values.
    pre_connect=(
        [
            PreConnectRequest(
                url=f"{PUBLIC_BASE_URL}/pre-connect/whois",
                returns=[Captured(name="customer_tier", path="customer_tier", default="standard")],
                timeout_ms=500,
                allow_overrides=True,  # a top-level "greeting" in the response replaces the greeting
            )
        ]
        if PUBLIC_BASE_URL
        else None
    ),
)
