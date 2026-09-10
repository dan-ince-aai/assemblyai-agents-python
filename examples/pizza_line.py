"""The agent declaration shared by the other examples.

Nothing in this module talks to the network, so it is safe to import from tests.

Every tool carries an ``http=`` config, built from ``PUBLIC_BASE_URL``. That is
the only arrangement that works on a phone call: the platform fetches the tool
itself, because there is no client on the line to ask. A pre-connect request is
declared the same way.
"""

import os

from assemblyai_agents import Captured, PreConnectRequest, VoiceAgent, tool
from assemblyai_agents.models.rest import (
    HttpMethod,
    HttpToolHeaderInput,
    LlmConfigRequest,
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


def hosted(path: str) -> PlaintextHttpToolConfig:
    """Where the platform fetches this tool from."""
    if not PUBLIC_BASE_URL:
        raise RuntimeError(
            "PUBLIC_BASE_URL is unset. Every tool is fetched over HTTPS, so the "
            "address of the process serving them has to be known before the "
            "agent is declared."
        )
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
    order = ORDERS.get(order_id.upper())
    if order is None:
        return {"error": f"no order {order_id}"}
    # Echoing the id back matters: whatever generates the reply reads the result
    # and nothing else, so a result that omits the id cannot name it out loud.
    return {"order_id": order_id.upper(), **order}


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


def byo_llm() -> LlmConfigRequest | None:
    """Point response generation at your own OpenAI-compatible endpoint, or None.

    `BYO_LLM=1` uses `byo_llm_server.py` on this same backend, which is what
    `e2e_check.py` exercises; `LLM_BASE_URL` points somewhere else entirely (a
    hosted model, a gateway, your own service). Unset both and the platform's
    default model runs the conversation.
    """
    base_url = os.environ.get("LLM_BASE_URL", "").rstrip("/")
    if not base_url and os.environ.get("BYO_LLM") and PUBLIC_BASE_URL:
        base_url = f"{PUBLIC_BASE_URL}/v1"
    if not base_url:
        return None
    return LlmConfigRequest(
        base_url=base_url,
        model=os.environ.get("LLM_MODEL", "pizza-line-rules"),
        api_key=os.environ.get("LLM_API_KEY", "change-me"),
    )


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
    # None unless BYO_LLM / LLM_BASE_URL is set, in which case your endpoint
    # generates every reply instead of the platform's default model.
    llm=byo_llm(),
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
