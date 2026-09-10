"""Read the platform's request, and answer it.

`llm=LlmConfigRequest(...)` points an agent at your own chat-completions
endpoint, and from then on the platform asks you what to say on every turn.
That is the seam to reach for when a script has to be exact, or when you want
your own logic and a model in the same call.

Between you and that seam is contract detail: Server-Sent Events, a transcript
whose shape is not obvious, and a platform that refuses tool calls carrying
values nobody said. This module is only that detail. How you decide what to say
is yours; there is no framework here.

    from assemblyai_agents.byo import Turn, call_tool, say, silence, stream

    def decide(turn):
        if turn.pending and turn.pending.name == "verify_caller":
            return say("Thanks, how can I help?") if turn.pending.get("verified") else say("Try again?")
        if not turn.caller_said:
            return say("Could you give me your full name?")
        return call_tool("verify_caller", caller_said=turn.caller_said)

    # on POST /v1/chat/completions, in whatever framework you use:
    turn = Turn.from_request(body)
    return StreamingResponse(stream(turn, decide(turn)), media_type="text/event-stream")

`examples/starter/` builds a whole agent on this, including one way of
organising `decide` as ordered stages. That organisation is an opinion, so it
lives in the example rather than here.
"""

import json
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional, Sequence

# The tool the platform runs itself before a phone call, whose result carries
# whatever the pre-connect requests captured.
PRE_CONNECT_TOOL = "aai_pre_connect_context"

_MARKER_LENGTH = 40


# --------------------------------------------------------------------------- what came back


@dataclass(frozen=True)
class ToolResult:
    """A tool call the platform has finished with, or refused to make.

    `ran` is the distinction that matters. A refused or failed call comes back
    as prose rather than the tool's own JSON, so a payment tool that was never
    reached must not be reported to the caller as a declined card.
    """

    name: str
    arguments: dict
    value: Any
    ran: bool
    note: str = ""

    def get(self, key: str, default: Any = None) -> Any:
        return self.value.get(key, default) if isinstance(self.value, dict) else default

    @property
    def keypad_incomplete(self) -> bool:
        """The caller did not finish entering a keypad field, so nothing ran."""
        return not self.ran and "keypad" in self.note


def _read_result(payload: Any) -> tuple:
    if not isinstance(payload, str):
        return payload, True, ""
    head = payload.split("\n[")[0]
    try:
        return json.loads(head), True, ""
    except ValueError:
        return None, False, head.strip()


# --------------------------------------------------------------------------- the turn


@dataclass(frozen=True)
class Turn:
    """One request from the platform, read.

    The call so far, in the shape you need to decide what to say next. Built
    with `Turn.from_request(body)`; nothing here touches the network.
    """

    request: dict
    messages: list = field(default_factory=list)
    tool_names: frozenset = frozenset()
    caller_said: str = ""
    spoken: tuple = ()
    preconnect: dict = field(default_factory=dict)
    pending: Optional[ToolResult] = None
    results: tuple = ()

    @classmethod
    def from_request(cls, body: dict) -> "Turn":
        messages = list(body.get("messages") or [])
        calls: dict = {}
        results: list = []
        spoken: list = []
        preconnect: dict = {}
        last_result_at = -1

        for index, message in enumerate(messages):
            for call in message.get("tool_calls") or []:
                function = call.get("function") or {}
                try:
                    arguments = json.loads(function.get("arguments") or "{}")
                except ValueError:
                    arguments = {}
                calls[call.get("id")] = (function.get("name"), arguments)
            if message.get("role") == "assistant" and message.get("content"):
                spoken.append(str(message["content"]))
            if message.get("role") != "tool":
                continue
            name, arguments = calls.get(message.get("tool_call_id"), ("", {}))
            value, ran, note = _read_result(message.get("content"))
            results.append(ToolResult(name=name, arguments=arguments, value=value, ran=ran, note=note))
            last_result_at = index
            if name == PRE_CONNECT_TOOL and isinstance(value, dict):
                # The platform ran the pre-connect requests itself and put what
                # they captured here. These count as values the call
                # established, so they are safe to pass to other tools.
                variables = value.get("variables")
                if isinstance(variables, dict):
                    preconnect = dict(variables)

        # A result is a cue to speak only while nothing has been said about it.
        # The tool message is usually not the last one: the platform adds its
        # own note once a call completes.
        pending = None
        if results and last_result_at >= 0 and results[-1].name != PRE_CONNECT_TOOL:
            answered = any(
                message.get("role") == "assistant" and message.get("content")
                for message in messages[last_result_at + 1 :]
            )
            if not answered:
                pending = results[-1]

        return cls(
            request=body,
            messages=messages,
            tool_names=frozenset(
                (tool.get("function") or {}).get("name")
                for tool in body.get("tools") or []
                if (tool.get("function") or {}).get("name")
            ),
            caller_said=_caller_said(messages),
            spoken=tuple(spoken),
            preconnect=preconnect,
            pending=pending,
            results=tuple(results),
        )

    def said_before(self, marker: str) -> bool:
        """Has the agent already said this?

        Matched on the opening of the line. Note that an interrupted turn does
        not always come back in the transcript, so a line the caller talked
        over can look unsaid. Keep your own note of anything that has to be
        said exactly once.
        """
        return marker.lower()[:_MARKER_LENGTH] in " ".join(self.spoken).lower()

    def answer_following(self, fragment: str) -> str:
        """What the caller said after the agent last said this.

        For a value collected over several turns: ask, then read the answer
        back out of the transcript rather than holding it in memory.
        """
        asked_at = None
        for index, message in enumerate(self.messages):
            if message.get("role") == "assistant" and fragment in str(message.get("content") or ""):
                asked_at = index
        if asked_at is None:
            return ""
        for message in self.messages[asked_at + 1 :]:
            if message.get("role") == "user" and isinstance(message.get("content"), str):
                return message["content"]
        return ""

    def result_of(self, name: str, arguments: Optional[dict] = None) -> Optional[ToolResult]:
        """The newest successful result from this tool, if the call has one.

        Reading an answer back out of the transcript is how you avoid asking
        the platform to run the same call twice, which the caller hears as the
        same silence twice.
        """
        for result in reversed(self.results):
            if result.name != name or not result.ran:
                continue
            if arguments is not None and result.arguments != arguments:
                continue
            return result
        return None

    def has(self, tool: str) -> bool:
        """Is this tool one the platform is offering on this turn?"""
        return tool in self.tool_names


def _caller_said(messages: Sequence[dict]) -> str:
    """The caller's most recent words.

    Speech arrives as a `user` message. A turn injected by a test driver
    through `create_reply(instructions=...)` arrives as a `system` message
    quoting it, while the platform's own notes use single quotes and so never
    match. A real `user` message always wins.
    """
    quoted = ""
    for message in reversed(list(messages)[1:]):
        content = message.get("content")
        if not isinstance(content, str):
            continue
        if message.get("role") == "user":
            return content
        if message.get("role") == "system" and not quoted:
            match = re.search(r'"([^"]{3,})"', content)
            if match:
                quoted = match.group(1)
    return quoted


# --------------------------------------------------------------------------- what to answer


@dataclass(frozen=True)
class Say:
    text: str


@dataclass(frozen=True)
class Silence:
    pass


@dataclass(frozen=True)
class Call:
    name: str
    arguments: dict


def say(text: str) -> Say:
    """Speak these words."""
    return Say(text)


def silence() -> Silence:
    """Say nothing.

    There is no way for an agent to hang up a phone call, so this is how a
    finished call ends: quiet, rather than looking for something else to ask.
    """
    return Silence()


def call_tool(name: str, **arguments: Any) -> Call:
    """Run one of the agent's tools.

    Arguments whose value is None or empty are dropped, because the platform
    refuses a call carrying a value the conversation never established and an
    empty string counts as invented. Pass the caller's own words and let the
    tool do the reading, or pass a value an earlier tool returned.
    """
    return Call(name, established(**arguments))


def established(**arguments: Any) -> dict:
    """Drop arguments the call has not established.

    The platform checks every argument against the conversation before running
    a tool and refuses the call otherwise, saying so: "the call has not
    established a value for X … Never invent a value."
    """
    return {name: value for name, value in arguments.items() if value not in (None, "")}


# --------------------------------------------------------------------------- answering


def stream(turn: Turn, answer: Any, *, chunk_words: bool = True) -> Iterator[str]:
    """The answer, as the Server-Sent Events the platform expects.

    Every request arrives with `stream: true`, so this is the only shape that
    works; a plain JSON body is not accepted. Words go one at a time by
    default, so speech starts before the sentence is finished.
    """
    completion_id = "chatcmpl-" + uuid.uuid4().hex[:16]
    created = int(time.time())
    model = turn.request.get("model", "byo")

    def frame(payload: dict) -> str:
        return f"data: {json.dumps(payload)}\n\n"

    def chunk(delta: dict, finish_reason: Optional[str] = None) -> dict:
        return {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }

    if isinstance(answer, Call):
        yield frame(chunk({"role": "assistant", "tool_calls": [_wire_call(answer)]}))
        yield frame(chunk({}, "tool_calls"))
    else:
        text = answer.text if isinstance(answer, Say) else ""
        yield frame(chunk({"role": "assistant", "content": ""}))
        for piece in (text.split(" ") if chunk_words else [text]) if text else []:
            yield frame(chunk({"content": piece + " " if chunk_words else piece}))
        yield frame(chunk({}, "stop"))

    if (turn.request.get("stream_options") or {}).get("include_usage"):
        yield frame({
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        })
    yield "data: [DONE]\n\n"


def json_body(turn: Turn, answer: Any) -> dict:
    """The same answer as a non-streaming body, for curl and for tests."""
    message: dict = {"role": "assistant", "content": answer.text if isinstance(answer, Say) else ""}
    finish = "stop"
    if isinstance(answer, Call):
        message = {"role": "assistant", "content": None, "tool_calls": [_wire_call(answer)]}
        finish = "tool_calls"
    return {
        "id": "chatcmpl-" + uuid.uuid4().hex[:16],
        "object": "chat.completion",
        "created": int(time.time()),
        "model": turn.request.get("model", "byo"),
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def _wire_call(answer: Call) -> dict:
    return {
        "index": 0,
        "id": "call_" + uuid.uuid4().hex[:16],
        "type": "function",
        "function": {"name": answer.name, "arguments": json.dumps(answer.arguments)},
    }


# --------------------------------------------------------------------------- speech


_UNITS = {
    "zero": "0", "oh": "0", "o": "0", "nought": "0", "one": "1", "won": "1",
    "two": "2", "to": "2", "too": "2", "three": "3", "four": "4", "for": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "ate": "8", "nine": "9",
}
_TEENS = {
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14",
    "fifteen": "15", "sixteen": "16", "seventeen": "17", "eighteen": "18", "nineteen": "19",
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fourty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}


def digits_said(text: str) -> str:
    """The digits in what a caller read out, however they read them.

    Reference numbers, card numbers, dates of birth and postcodes all arrive as
    a mix of figures and words, grouped however speech to text felt like it:
    "four four seven one", "forty one eleven", "double one". Every token that
    carries digits is kept in the order it was said and the rest is dropped,
    which is enough for a value you then validate.

        digits_said("it's four four seven one")   -> "4471"
        digits_said("forty one eleven")           -> "4111"
        digits_said("4 1 double 1")               -> "4111"
    """
    out: list = []
    repeat = 1
    pending_tens: Optional[int] = None
    for token in re.findall(r"\d+|[a-z]+", (text or "").lower()):
        if token in ("double", "triple"):
            repeat = 2 if token == "double" else 3
            continue
        if token in _TENS:
            # "forty one" is 41, not 40 then 1, so a tens word waits for a unit.
            if pending_tens is not None:
                out.append(str(pending_tens))
            pending_tens = _TENS[token]
            continue
        piece = token if token.isdigit() else _UNITS.get(token) or _TEENS.get(token)
        if piece is None:
            continue
        if pending_tens is not None:
            if len(piece) == 1 and piece != "0":
                out.append(str(pending_tens + int(piece)))
                pending_tens = None
                repeat = 1
                continue
            out.append(str(pending_tens))
            pending_tens = None
        out.append(piece * repeat)
        repeat = 1
    if pending_tens is not None:
        out.append(str(pending_tens))
    return "".join(out)
