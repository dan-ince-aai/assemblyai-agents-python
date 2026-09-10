"""Bring your own LLM: your endpoint generates the agent's replies.

This is `server.py` (tools + webhooks) plus one more route,
``POST /v1/chat/completions``, so a single service owns the whole conversation:
the platform asks *your* endpoint what to say on every turn, and your endpoint
can ask the platform to run the agent's tools.

    pip install fastapi uvicorn
    export TOOL_SECRET=... LLM_API_KEY=...
    uvicorn byo_llm_server:app --port 8000       # run from the examples/ directory

Expose it over HTTPS, then deploy the agent with the LLM wired to it:

    BYO_LLM=1 PUBLIC_BASE_URL=https://<public host> TOOL_SECRET=... LLM_API_KEY=... python deploy.py

The responder here is deliberately not a model at all - it is a few lines of
Python that read the transcript and decide. Swap ``decide()`` for a call to
your own model, an open-weights model you host, a decision tree, or a
retrieval-augmented pipeline; the platform cannot tell the difference, because
the seam is the OpenAI chat-completions schema.

What the platform sends (captured from a live session):

* ``POST {base_url}/chat/completions`` with ``Authorization: Bearer <api_key>``
  (the key from ``LlmConfigRequest``) and ``User-Agent: LiveKit Agents/...``.
* ``stream: true`` always, with ``stream_options: {"include_usage": true}``, so
  the endpoint has to answer with Server-Sent Events.
* ``model`` is the string from ``LlmConfigRequest.model`` - your endpoint can
  ignore it or route on it.
* ``tools`` carries the agent's tools in OpenAI function form, plus
  ``tool_choice: "auto"``. Note the platform nests a second ``type: "function"``
  and its own ``timeout_seconds``/``execution_mode`` inside ``function``, so
  read the name from ``tool["function"]["name"]``.
* ``messages`` starts with your ``system_prompt`` **plus the platform's own
  spoken-output guidance** appended (formatting, how to say numbers aloud, when
  to call tools). The greeting appears as an ``assistant`` message.
* Emit ``tool_calls`` and the platform runs the tool, then calls you again with
  a ``tool`` message carrying the result and ``tool_call_id``. A failure comes
  back as a ``tool`` message with coaching text, and after three consecutive
  failures the platform tells you to stop retrying.
* The client sends a read timeout of 10 seconds, so get the first SSE chunk out
  quickly; do slow work in a tool instead.
"""

import json
import os
import re
import time
import uuid

from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse

from server import app  # tools + webhook routes, and the auth check they use

LLM_API_KEY = os.environ.get("LLM_API_KEY", "change-me")
# Set BYO_LLM_DUMP=/path/to/file.jsonl to append every request body, which is
# the quickest way to see what the platform actually sends on each turn.
DUMP_PATH = os.environ.get("BYO_LLM_DUMP")

ORDER_CODE = re.compile(r"\b([wW]\s*\d{3,}|[wW]-?\d{3,})\b")


# --------------------------------------------------------------------------- the "model"


def spoken_order_id(text: str) -> str | None:
    """Pull an order number out of what the caller said, e.g. "order W004"."""
    match = ORDER_CODE.search(text)
    return match.group(1).upper().replace(" ", "").replace("-", "") if match else None


def caller_line(messages: list[dict]) -> str:
    """What the caller most recently said.

    Real speech arrives as a ``user`` message. A turn injected with
    ``create_reply(instructions=...)`` arrives as a ``system`` message, and so
    do the platform's own nudges ("the function call … encountered an error",
    "a function is already executing"). Prefer a real user message, and fall
    back to text the last system message quotes, so a platform nudge is never
    mistaken for something the caller said.
    """
    quoted_line = ""
    for message in reversed(messages[1:]):
        content = message.get("content")
        if not isinstance(content, str):
            continue
        if message.get("role") == "user":
            return content
        if message.get("role") == "system" and not quoted_line:
            # An injected turn quotes the caller; the platform's own nudges use
            # single quotes around arguments, so they do not match.
            quoted = re.search(r'"([^"]{4,})"', content)
            if quoted:
                quoted_line = quoted.group(1)
    return quoted_line


def result_already_in(messages: list[dict], name: str, arguments: dict) -> dict | None:
    """The result of an identical earlier call, if this conversation has one.

    The platform executes every ``tool_calls`` it is given, so a responder that
    keeps asking for the same call keeps the caller waiting through the same
    round trip. Its own guidance says as much: don't call a tool again with the
    same arguments. Reading the answer back out of the transcript is how you
    honour that.
    """
    pending: dict[str, tuple[str, dict]] = {}
    for message in messages:
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            try:
                parsed = json.loads(function.get("arguments") or "{}")
            except ValueError:
                parsed = {}
            pending[call.get("id")] = (function.get("name"), parsed)
        if message.get("role") == "tool":
            called = pending.get(message.get("tool_call_id"))
            if called and called == (name, arguments):
                return parse_tool_result(message.get("content"))
    return None


def parse_tool_result(payload) -> dict:
    """The tool's JSON body, as the platform hands it back.

    A failed call arrives as the body followed by bracketed coaching text, so
    only the part before that is JSON.
    """
    if not isinstance(payload, str):
        return {"error": "no result"}
    try:
        return json.loads(payload.split("\n[")[0])
    except ValueError:
        return {"error": payload[:200]}


def answer_from(result: dict) -> str:
    """Turn a tool result into a sentence, ready to be spoken."""
    if not isinstance(result, dict) or result.get("error"):
        return "I could not find that order. Could you read the number out again?"
    if result.get("cancelled") is True:
        return "That is cancelled now. Anything else I can do?"
    if result.get("cancelled") is False:
        return f"I cannot cancel that one: {result.get('reason', 'it is too late')}."
    if "status" in result:
        # The platform's guidance says to spell identifiers out digit by digit,
        # because this text is spoken aloud.
        order_id = result.get("order_id")
        spoken = " ".join(str(order_id)) if order_id else "that order"
        eta = result.get("eta")
        return f"Order {spoken} is {result['status']}" + (f", expected {eta}." if eta else ".")
    return "That is all sorted. Anything else?"


def plan(said: str, tool_names: set[str]) -> dict | None:
    """Which tool, if any, this line calls for."""
    order_id = spoken_order_id(said)
    if not order_id:
        return None
    if "cancel" in said.lower() and "cancel_order" in tool_names:
        return tool_call("cancel_order", {"order_id": order_id, "reason": "caller asked on the phone"})
    if "lookup_order" in tool_names:
        return tool_call("lookup_order", {"order_id": order_id})
    return None


def unanswered_tool_result(messages: list[dict]) -> dict | None:
    """The newest tool result that has not been spoken about yet.

    The tool message is often not the last one: the platform appends its own
    system note after a call, and the turn that carried the caller's words may
    no longer be in the transcript at all. So the cue to answer is a tool result
    with no assistant *text* after it, rather than a tool message in last place.
    """
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if message.get("role") == "tool":
            spoken_since = any(
                later.get("role") == "assistant" and later.get("content")
                for later in messages[index + 1 :]
            )
            return None if spoken_since else parse_tool_result(message.get("content"))
    return None


def decide(messages: list[dict], tool_names: set[str]) -> tuple[str | None, dict | None]:
    """Return either ``(text, None)`` to speak, or ``(None, tool_call)`` to call a tool.

    This is the whole "LLM". Replace it with whatever you want - your own model,
    an open-weights model you host, a decision tree, a retrieval pipeline -
    and everything around it stays as it is.
    """
    # A tool the platform ran for us has come back: answer from its result.
    pending = unanswered_tool_result(messages)
    if pending is not None:
        return answer_from(pending), None

    said = caller_line(messages)
    candidate = plan(said, tool_names)
    if candidate is not None:
        function = candidate["function"]
        arguments = json.loads(function["arguments"])
        earlier = result_already_in(messages, function["name"], arguments)
        if earlier is not None:
            return answer_from(earlier), None
        return None, candidate

    lowered = said.lower()
    if "cancel" in lowered:
        return "Sure, I can cancel that. What is the order number?", None
    if any(word in lowered for word in ("order", "delivery", "status", "where")):
        return "Happy to check. What is the order number?", None
    return "You have reached Pizza Palace. I can check an order or cancel one. Which would you like?", None


def tool_call(name: str, arguments: dict) -> dict:
    return {
        "id": "call_" + uuid.uuid4().hex[:16],
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


# --------------------------------------------------------------------------- the endpoint


def tool_names_from(body: dict) -> set[str]:
    # The platform nests a second `type` and its own fields inside `function`,
    # so read the name from there rather than from the outer object.
    return {
        (tool.get("function") or {}).get("name")
        for tool in body.get("tools") or []
        if (tool.get("function") or {}).get("name")
    }


def sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def stream_reply(body: dict, text: str | None, call: dict | None):
    """Server-Sent Events in OpenAI's streaming chat-completions shape."""
    completion_id = "chatcmpl-" + uuid.uuid4().hex[:16]
    created = int(time.time())
    model = body.get("model", "byo")

    def chunk(delta: dict, finish_reason: str | None = None) -> dict:
        return {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }

    def generate():
        if call is not None:
            # A streamed tool call: index identifies it across chunks, and the
            # arguments may be split over several - sending them whole is fine.
            yield sse(chunk({"role": "assistant", "tool_calls": [{"index": 0, **call}]}))
            yield sse(chunk({}, "tool_calls"))
        else:
            yield sse(chunk({"role": "assistant", "content": ""}))
            # Sent word by word so speech starts before the sentence is finished.
            for word in (text or "").split(" "):
                yield sse(chunk({"content": word + " "}))
            yield sse(chunk({}, "stop"))
        if (body.get("stream_options") or {}).get("include_usage"):
            yield sse({
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            })
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


def whole_reply(body: dict, text: str | None, call: dict | None) -> JSONResponse:
    """The non-streaming form. The platform always streams; this is for curl."""
    message = {"role": "assistant", "content": text}
    if call is not None:
        message = {"role": "assistant", "content": None, "tool_calls": [call]}
    return JSONResponse({
        "id": "chatcmpl-" + uuid.uuid4().hex[:16],
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.get("model", "byo"),
        "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if call else "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    })


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    if request.headers.get("Authorization") != f"Bearer {LLM_API_KEY}":
        return JSONResponse({"error": {"message": "bad api key", "type": "invalid_request_error"}}, status_code=401)
    body = await request.json()
    if DUMP_PATH:
        with open(DUMP_PATH, "a") as dump:
            dump.write(json.dumps({"at": time.time(), "body": body}) + "\n")
    messages = body.get("messages") or []
    names = tool_names_from(body)
    text, call = decide(messages, names)
    decision = (
        f"tool_call {call['function']['name']}({call['function']['arguments']})"
        if call
        else f"say {text!r}"
    )
    # flush=True: uvicorn buffers stdout, and a decision log you cannot see
    # while a call is in flight is no use.
    print(f"[llm] roles={[m.get('role') for m in messages]} tools={sorted(names)} -> {decision}", flush=True)
    if body.get("stream"):
        return stream_reply(body, text, call)
    return whole_reply(body, text, call)


@app.get("/v1/models")
def models():
    """Not required by the platform; handy for pointing other OpenAI clients here."""
    return {"object": "list", "data": [{"id": "pizza-line-rules", "object": "model", "owned_by": "you"}]}
