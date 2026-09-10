"""Write the agent's replies yourself, without writing the plumbing.

`llm=LlmConfigRequest(...)` points an agent at your own chat-completions
endpoint, and from then on the platform asks *you* what to say on every turn.
That is the seam to reach for when a script has to be exact: a required
disclosure, a regulated order of steps, an amount that must come from a ledger
rather than from a sentence. It is also the seam for mixing the two, with
deterministic stages where the wording matters and a model where the caller's
words do.

What sits between you and that is a pile of contract detail: Server-Sent
Events, a transcript whose shape is not obvious, and a platform that refuses
tool calls carrying values nobody said. This module is that pile, so a reply
engine can be the part you actually care about:

    from assemblyai_agents.byo import Call, Say, Stages, Turn, sse

    stages = Stages()

    @stages.stage("verify", until=lambda turn: turn.result_of("verify_caller"))
    def verify(turn):
        if not turn.caller_said:
            return Say("Could you give me your full name?")
        return Call("verify_caller", caller_said=turn.caller_said)

    @stages.stage("business")
    def business(turn):
        return Say("Thanks, how can I help?")

    # in your web framework, on POST /v1/chat/completions:
    turn = Turn.from_request(body)
    return sse(turn, stages.decide(turn))     # an iterator of SSE lines

Everything here is plain data and iterators, so it drops into FastAPI, Flask,
Django or anything else without this module knowing which.
"""

import json
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Optional, Sequence

# The tool the platform runs itself before a phone call, whose result carries
# whatever the pre-connect requests captured.
PRE_CONNECT_TOOL = "aai_pre_connect_context"

# How much of a line has to match for `said_before` to call it said.
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


def _parse_result(payload: Any) -> tuple[Any, bool, str]:
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

    Built with `Turn.from_request(body)`. Nothing here talks to the network.
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
        last_result_index = -1

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
            value, ran, note = _parse_result(message.get("content"))
            result = ToolResult(name=name, arguments=arguments, value=value, ran=ran, note=note)
            results.append(result)
            last_result_index = index
            if name == PRE_CONNECT_TOOL and isinstance(value, dict):
                # The platform ran the pre-connect requests itself and put what
                # they captured here. These count as values the call
                # established, so they are safe to pass to other tools.
                variables = value.get("variables")
                if isinstance(variables, dict):
                    preconnect = dict(variables)

        # A result is only a cue to speak while nothing has been said about it.
        # The tool message is often not the last one: the platform adds its own
        # note after a call completes.
        pending = None
        if results and last_result_index >= 0:
            spoken_since = any(
                message.get("role") == "assistant" and message.get("content")
                for message in messages[last_result_index + 1 :]
            )
            if not spoken_since and results[-1].name != PRE_CONNECT_TOOL:
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

    # -- reading the call ---------------------------------------------------

    def said_before(self, marker: str) -> bool:
        """Has the agent already said this?

        Matched on the opening of the line. Note that an interrupted turn does
        not always come back in the transcript, so a line the caller talked
        over can look unsaid: keep your own note of anything that must be said
        exactly once.
        """
        joined = " ".join(self.spoken).lower()
        return marker.lower()[:_MARKER_LENGTH] in joined

    def answer_following(self, fragment: str) -> str:
        """What the caller said after the agent last said this.

        For collecting a value over several turns: ask, then read the answer
        back out of the transcript rather than holding it in memory.
        """
        asked_at = None
        for index, message in enumerate(self.messages):
            if message.get("role") == "assistant" and fragment in str(message.get("content") or ""):
                asked_at = index
        if asked_at is None:
            return ""
        for message in self.messages[asked_at + 1 :]:
            content = message.get("content")
            if isinstance(content, str) and message.get("role") == "user":
                return content
        return ""

    def result_of(self, name: str, arguments: Optional[dict] = None) -> Optional[ToolResult]:
        """The newest successful result from this tool, if the call has one.

        Reading an answer back out of the transcript is how you avoid asking
        the platform to run the same call twice, which the caller experiences
        as the same silence twice.
        """
        for result in reversed(self.results):
            if result.name != name or not result.ran:
                continue
            if arguments is not None and result.arguments != arguments:
                continue
            return result
        return None

    def has(self, tool: str) -> bool:
        return tool in self.tool_names

    # -- answering ----------------------------------------------------------

    def say(self, text: str) -> "Say":
        """Speak these words."""
        return Say(text)

    def call(self, name: str, **arguments: Any) -> "Call":
        """Run a tool. Arguments the call has not established are dropped."""
        return Call(name, **arguments)

    def silence(self) -> "Silence":
        """Say nothing, which is how a finished call ends."""
        return Silence()


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


# --------------------------------------------------------------------------- what to do next


@dataclass(frozen=True)
class Say:
    """Speak these words."""

    text: str


@dataclass(frozen=True)
class Silence:
    """Say nothing.

    There is no way for an agent to hang up a phone call, so this is how a
    finished call ends: quiet, rather than looking for something else to ask.
    """


@dataclass(frozen=True)
class Call:
    """Run a tool.

    Arguments whose value is None or empty are dropped, because the platform
    refuses a call carrying a value the conversation never established and an
    empty string counts as invented. Pass the caller's own words and let the
    tool do the reading, or pass a value an earlier tool returned.
    """

    name: str
    arguments: dict

    def __init__(self, name: str, **arguments: Any) -> None:
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "arguments", established(**arguments))


Action = Any  # Say | Silence | Call


def established(**arguments: Any) -> dict:
    """Drop arguments the call has not established.

    The platform checks every argument against the conversation before running
    a tool and refuses the call otherwise, with a note that says so: "the call
    has not established a value for X … Never invent a value."
    """
    return {name: value for name, value in arguments.items() if value not in (None, "")}


# --------------------------------------------------------------------------- stages


@dataclass
class Stage:
    name: str
    handler: Callable[[Turn], Action]
    until: Optional[Callable[[Turn], Any]] = None


class Responder:
    """The thing that answers: ordered stages, and the wire format handled.

    A call has a shape, and each part wants different machinery. The ends are
    usually a script; the middle is a conversation. One handler trying to be
    both becomes a tangle of conditions, so each stage is its own small agent:
    it owns a few turns and hands over when its own test says it is done.

    A call has a shape: identify, disclose, transact, close. Each part wants
    different machinery — the middle of a call is a conversation, the ends are
    usually a script — and a single handler that tries to be both becomes a
    tangle of conditions. So each stage is its own small agent: it owns a few
    turns, and hands over when it is done.

        stages = Stages()

        @stages.stage("identify", until=lambda turn: verified(turn))
        def identify(turn): ...

        @stages.stage("collect", until=lambda turn: turn.result_of("take_payment"))
        def collect(turn): ...

        @stages.stage("close")            # no `until`: the last one
        def close(turn): ...

    The first stage whose `until` has not yet been satisfied handles the turn.
    A stage returns `turn.say(...)`, `turn.call(...)` or `turn.silence()`, or
    None to fall through to the next one.

    Then either hand it the request body:

        reply = responder.respond(body)
        return StreamingResponse(reply.stream(), media_type=reply.media_type)

    or let it mount itself, which also serves the tools:

        mount_fastapi(app, responder, tools=TOOLS, tool_secret=..., llm_key=...)
    """

    def __init__(self) -> None:
        self._stages: list = []

    def stage(self, name: str, until: Optional[Callable[[Turn], Any]] = None) -> Callable:
        def register(handler: Callable[[Turn], Action]) -> Callable[[Turn], Action]:
            self._stages.append(Stage(name=name, handler=handler, until=until))
            return handler

        return register

    def add(self, name: str, handler: Callable, until: Optional[Callable] = None) -> None:
        self._stages.append(Stage(name=name, handler=handler, until=until))

    def current(self, turn: Turn) -> Optional[Stage]:
        for stage in self._stages:
            if stage.until is None or not stage.until(turn):
                return stage
        return None

    def decide(self, turn: Turn) -> Action:
        """Ask each stage in turn until one answers."""
        for stage in self._stages:
            if stage.until is not None and stage.until(turn):
                continue
            action = stage.handler(turn)
            if action is not None:
                return action
        return Silence()

    def respond(self, body: dict) -> "Reply":
        """Read the request, decide, and hand back something sendable."""
        turn = Turn.from_request(body)
        return Reply(turn=turn, action=self.decide(turn), stage=self.current(turn))

    @property
    def names(self) -> tuple:
        return tuple(stage.name for stage in self._stages)


# Kept as the older name for this surface.
Stages = Responder


@dataclass(frozen=True)
class Reply:
    """A decided answer, in whichever shape the caller needs.

    Every request from the platform arrives with `stream: true`, so `stream()`
    is the one that matters; `json()` is for curl and for tests.
    """

    turn: Turn
    action: Action
    stage: Optional[Stage] = None
    media_type: str = "text/event-stream"

    def stream(self, *, chunk_words: bool = True) -> Iterator[str]:
        return sse(self.turn, self.action, chunk_words=chunk_words)

    def json(self) -> dict:
        return completion(self.turn, self.action)

    @property
    def spoken(self) -> str:
        return self.action.text if isinstance(self.action, Say) else ""

    @property
    def tool(self) -> Optional[str]:
        return self.action.name if isinstance(self.action, Call) else None

    def __str__(self) -> str:
        where = f"[{self.stage.name}] " if self.stage else ""
        if isinstance(self.action, Call):
            return f"{where}call {self.action.name}({json.dumps(self.action.arguments)})"
        if isinstance(self.action, Silence):
            return f"{where}silence"
        return f"{where}say {self.spoken!r}"


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
    a mix of figures and words, and speech to text groups them however it
    likes: "four four seven one", "forty one eleven", "double one". Every token
    that carries digits is kept in the order it was said and the rest is
    dropped, which is enough for a value you then validate.

        digits_said("it's four four seven one")     -> "4471"
        digits_said("4 1 double 1 then 1111")       -> "411111111"
        digits_said("forty one eleven")             -> "4111"
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


# --------------------------------------------------------------------------- remembering


class Memo:
    """A note of what has already happened on a call.

    Nearly everything a reply engine needs can be read back out of the
    transcript, and should be. One thing cannot: whether a particular line has
    actually been spoken. A caller who talks over a long scripted sentence
    leaves that turn interrupted, and it does not come back in the messages, so
    a check of "have I said this yet" answers no and the line starts again.
    Anything that must be said exactly once needs a note kept here.

        memo = Memo()
        ...
        if not memo.has(call_key, "notice"):
            memo.note(call_key, "notice")
            return Say(NOTICE)

    Process-local, so one worker holds it. A fleet would keep the same notes
    wherever it keeps session state, and `forget` at the start of each call.
    """

    def __init__(self) -> None:
        self._notes: dict = {}

    def note(self, key: str, *markers: str) -> None:
        self._notes.setdefault(key or "-", set()).update(markers)

    def has(self, key: str, marker: str) -> bool:
        return marker in self._notes.get(key or "-", set())

    def forget(self, key: Optional[str] = None) -> None:
        if key is None:
            self._notes.clear()
        else:
            self._notes.pop(key, None)


# --------------------------------------------------------------------------- answering


def sse(turn: Turn, action: Action, *, chunk_words: bool = True) -> Iterator[str]:
    """The action, as the Server-Sent Events the platform expects.

    Every request arrives with `stream: true`, so this is the only shape that
    works; a plain JSON body is not accepted. Words are sent one at a time by
    default so speech starts before the sentence is finished.
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

    if isinstance(action, Call):
        call = {
            "index": 0,
            "id": "call_" + uuid.uuid4().hex[:16],
            "type": "function",
            "function": {"name": action.name, "arguments": json.dumps(action.arguments)},
        }
        yield frame(chunk({"role": "assistant", "tool_calls": [call]}))
        yield frame(chunk({}, "tool_calls"))
    else:
        text = action.text if isinstance(action, Say) else ""
        yield frame(chunk({"role": "assistant", "content": ""}))
        if text:
            for piece in (text.split(" ") if chunk_words else [text]):
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


# --------------------------------------------------------------------------- serving tools


def tool_runner(tools: Sequence[Any]) -> Callable:
    """A callable that runs one of your tools by name.

    Wraps the two failure modes worth distinguishing: a name that is not a tool
    of this agent, and arguments that do not fit its signature.

        run = tool_runner(TOOLS)
        result = await run("verify_caller", {"caller_said": "..."})
    """
    by_name = {declared.name: declared for declared in tools}

    async def run(name: str, arguments: dict) -> Any:
        declared = by_name.get(name)
        if declared is None:
            raise LookupError(f"no tool named {name!r}")
        try:
            return await declared.invoke(**arguments)
        except TypeError as exc:
            raise ValueError(str(exc)) from exc

    run.names = tuple(sorted(by_name))  # type: ignore[attr-defined]
    return run


def mount_fastapi(
    app: Any,
    responder: "Responder",
    *,
    tools: Sequence[Any] = (),
    tool_secret: str = "",
    llm_key: str = "",
    pre_connect: Optional[dict] = None,
    webhook_secret: Optional[str] = None,
    prefix: str = "",
    log: Optional[Callable[[str], None]] = print,
) -> Any:
    """Put every surface the platform calls onto a FastAPI app.

        mount_fastapi(app, responder, tools=TOOLS, tool_secret=..., llm_key=...,
                      pre_connect={"/pre-connect/lookup": lookup},
                      webhook_secret=...)

    Adds `POST /v1/chat/completions`, `POST /tools/{name}`, one route per
    pre-connect handler, `POST /webhooks/voice-agents` when a secret is given,
    and `GET /healthz`. Each is a few lines you could write yourself; having
    them here means the auth check and the streaming shape are not written
    again in every project.

    FastAPI is imported here rather than at module scope, so this package does
    not depend on it.
    """
    from fastapi import HTTPException, Request, Response  # noqa: PLC0415
    from fastapi.responses import JSONResponse, StreamingResponse  # noqa: PLC0415

    run_tool = tool_runner(tools)

    def note(message: str) -> None:
        if log is not None:
            log(message)

    @app.post(f"{prefix}/v1/chat/completions")
    async def _replies(request: Request):  # pragma: no cover - thin glue
        if llm_key and request.headers.get("Authorization") != f"Bearer {llm_key}":
            return JSONResponse({"error": {"message": "bad api key"}}, status_code=401)
        body = await request.json()
        reply = responder.respond(body)
        note(f"[reply] {reply}")
        if body.get("stream"):
            return StreamingResponse(reply.stream(), media_type=reply.media_type)
        return JSONResponse(reply.json())

    @app.post(f"{prefix}/tools/{{name}}")
    async def _tools(name: str, request: Request):  # pragma: no cover - thin glue
        if tool_secret and request.headers.get("Authorization") != f"Bearer {tool_secret}":
            raise HTTPException(status_code=401, detail="unauthorized")
        arguments = await request.json()
        note(f"[tool] {name}({json.dumps(arguments)[:120]})")
        try:
            result = await run_tool(name, arguments)
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        note(f"[tool] {name} -> {json.dumps(result, default=str)[:160]}")
        return result

    for path, handler in (pre_connect or {}).items():
        def _make(handler=handler, path=path):
            async def _pre_connect(request: Request):  # pragma: no cover - thin glue
                if tool_secret and request.headers.get("Authorization") != f"Bearer {tool_secret}":
                    raise HTTPException(status_code=401, detail="unauthorized")
                raw = await request.body()
                payload = json.loads(raw) if raw else {}
                result = handler(payload)
                if hasattr(result, "__await__"):
                    result = await result
                note(f"[pre-connect] {path} -> {json.dumps(result, default=str)[:160]}")
                return result

            return _pre_connect

        app.post(f"{prefix}{path}")(_make())

    if webhook_secret:
        from . import webhooks as _webhooks  # noqa: PLC0415

        @app.post(f"{prefix}/webhooks/voice-agents")
        async def _webhook(request: Request):  # pragma: no cover - thin glue
            body = await request.body()
            try:
                event = _webhooks.verify(
                    body, request.headers.get("X-AAI-Signature", ""), webhook_secret
                )
            except Exception as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            note(f"[webhook] {event.get('event') or event.get('type')}")
            return Response(status_code=204)

    @app.get(f"{prefix}/healthz")
    def _health():  # pragma: no cover - thin glue
        return {"ok": True, "stages": list(responder.names), "tools": list(run_tool.names)}

    return app


def completion(turn: Turn, action: Action) -> dict:
    """The same action as a non-streaming body, for curl and for tests."""
    message: dict = {"role": "assistant", "content": action.text if isinstance(action, Say) else ""}
    finish = "stop"
    if isinstance(action, Call):
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_" + uuid.uuid4().hex[:16],
                "type": "function",
                "function": {"name": action.name, "arguments": json.dumps(action.arguments)},
            }],
        }
        finish = "tool_calls"
    return {
        "id": "chatcmpl-" + uuid.uuid4().hex[:16],
        "object": "chat.completion",
        "created": int(time.time()),
        "model": turn.request.get("model", "byo"),
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
