"""The bring-your-own-reply-engine surface.

Every case here is a shape the platform actually sends, or a trap it sets. The
comments say which, because that is the part a reader cannot guess.
"""

import json

import pytest
from assemblyai_agents.byo import (
    Call,
    Memo,
    Reply,
    Responder,
    Say,
    Silence,
    Turn,
    completion,
    digits_said,
    established,
    sse,
    tool_runner,
)


def request(messages, tools=("do_thing",), **extra):
    return {
        "model": "engine",
        "stream": True,
        "stream_options": {"include_usage": True},
        "tools": [
            {"type": "function", "function": {"type": "function", "name": name, "parameters": {}}}
            for name in tools
        ],
        "messages": messages,
        **extra,
    }


def tool_call(call_id, name, arguments):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
        ],
    }


# --------------------------------------------------------------------------- reading a turn


def test_the_callers_words_come_from_a_user_message():
    turn = Turn.from_request(
        request([
            {"role": "system", "content": "prompt"},
            {"role": "assistant", "content": "Hello"},
            {"role": "user", "content": "I'd like to book"},
        ])
    )
    assert turn.caller_said == "I'd like to book"
    assert turn.spoken == ("Hello",)


def test_a_turn_injected_by_a_test_driver_is_read_from_the_quoted_instruction():
    # `create_reply(instructions=...)` arrives as a system message quoting the
    # caller. The platform's own notes use single quotes, so they never match.
    turn = Turn.from_request(
        request([
            {"role": "system", "content": "prompt"},
            {"role": "assistant", "content": "Hello"},
            {"role": "system", "content": 'The caller just said: "book me in". Respond.'},
        ])
    )
    assert turn.caller_said == "book me in"


def test_a_platform_note_is_not_mistaken_for_something_the_caller_said():
    turn = Turn.from_request(
        request([
            {"role": "system", "content": "prompt"},
            {"role": "assistant", "content": "Hello"},
            {"role": "system", "content": "The function call do_thing(id='7') has just completed."},
        ])
    )
    assert turn.caller_said == ""


def test_pre_connect_captures_are_read_from_the_tool_result_the_platform_injects():
    # On a phone call the platform runs the pre-connect requests itself and puts
    # what they captured in the transcript under this tool.
    turn = Turn.from_request(
        request([
            {"role": "system", "content": "prompt"},
            tool_call("pc", "aai_pre_connect_context", {}),
            {"role": "tool", "tool_call_id": "pc", "content": json.dumps({"variables": {"reference": "4471"}})},
            {"role": "assistant", "content": "Hello"},
        ])
    )
    assert turn.preconnect == {"reference": "4471"}
    # It is context, not something to answer, so it is never the pending cue.
    assert turn.pending is None


def test_a_finished_tool_is_pending_until_something_has_been_said_about_it():
    messages = [
        {"role": "system", "content": "prompt"},
        tool_call("c1", "do_thing", {"id": "7"}),
        {"role": "tool", "tool_call_id": "c1", "content": json.dumps({"ok": True})},
        # The platform adds its own note after a call completes, so the tool
        # message is usually not the last one.
        {"role": "system", "content": "The function call do_thing has just completed."},
    ]
    turn = Turn.from_request(request(messages))
    assert turn.pending is not None
    assert turn.pending.name == "do_thing"
    assert turn.pending.get("ok") is True

    spoken_since = Turn.from_request(request(messages + [{"role": "assistant", "content": "Done"}]))
    assert spoken_since.pending is None


def test_a_refused_call_is_told_apart_from_a_result():
    # A refusal comes back as prose with bracketed coaching text appended, not
    # as the tool's JSON. Reporting it as a result would tell the caller
    # something that never happened.
    refusal = (
        "The caller did not finish entering 'card_number' on their keypad (too_short), so the "
        "tool was not called.\n[The tool did not run, so nothing has changed.]"
    )
    turn = Turn.from_request(
        request([
            {"role": "system", "content": "prompt"},
            tool_call("c1", "take_payment", {"amount": 10}),
            {"role": "tool", "tool_call_id": "c1", "content": refusal},
        ])
    )
    assert turn.pending.ran is False
    assert turn.pending.keypad_incomplete is True
    assert "too_short" in turn.pending.note


def test_an_earlier_result_can_be_read_back_instead_of_calling_again():
    turn = Turn.from_request(
        request([
            {"role": "system", "content": "prompt"},
            tool_call("c1", "do_thing", {"id": "7"}),
            {"role": "tool", "tool_call_id": "c1", "content": json.dumps({"value": 42})},
            {"role": "assistant", "content": "Right"},
            {"role": "user", "content": "and again?"},
        ])
    )
    assert turn.result_of("do_thing").get("value") == 42
    assert turn.result_of("do_thing", {"id": "8"}) is None
    assert turn.result_of("missing") is None


def test_a_value_collected_over_turns_is_read_back_out_of_the_transcript():
    turn = Turn.from_request(
        request([
            {"role": "system", "content": "prompt"},
            {"role": "assistant", "content": "What is your date of birth?"},
            {"role": "user", "content": "third of March ninety two"},
            {"role": "assistant", "content": "And your postcode?"},
            {"role": "user", "content": "SW1A 1AA"},
        ])
    )
    assert turn.answer_following("date of birth") == "third of March ninety two"
    assert turn.answer_following("postcode") == "SW1A 1AA"
    assert turn.answer_following("never asked") == ""
    assert turn.said_before("What is your date of birth")


# --------------------------------------------------------------------------- deciding


def test_a_tool_call_drops_arguments_the_call_has_not_established():
    # The platform refuses a call carrying a value nobody said, and an empty
    # string counts as invented.
    action = Call("verify", caller_said="it's Maria", reference="", note=None, count=0)
    assert action.arguments == {"caller_said": "it's Maria", "count": 0}
    assert established(a="x", b="", c=None) == {"a": "x"}


def test_the_turn_verbs_build_the_same_actions():
    turn = Turn.from_request(request([{"role": "system", "content": "p"}]))
    assert turn.say("hello") == Say("hello")
    assert turn.call("do_thing", id="7").arguments == {"id": "7"}
    assert isinstance(turn.silence(), Silence)


def test_stages_hand_over_when_their_own_test_is_satisfied():
    responder = Responder()
    seen = []

    @responder.stage("identify", until=lambda turn: turn.result_of("verify") is not None)
    def identify(turn):
        seen.append("identify")
        return turn.call("verify", caller_said=turn.caller_said)

    @responder.stage("business")
    def business(turn):
        seen.append("business")
        return turn.say("How can I help?")

    first = Turn.from_request(request([{"role": "system", "content": "p"}, {"role": "user", "content": "hi"}]))
    assert responder.current(first).name == "identify"
    assert isinstance(responder.decide(first), Call)

    after = Turn.from_request(
        request([
            {"role": "system", "content": "p"},
            tool_call("c1", "verify", {"caller_said": "hi"}),
            {"role": "tool", "tool_call_id": "c1", "content": json.dumps({"verified": True})},
            {"role": "assistant", "content": "Thanks"},
        ])
    )
    assert responder.current(after).name == "business"
    assert responder.decide(after) == Say("How can I help?")
    assert seen == ["identify", "business"]


def test_a_stage_that_returns_nothing_falls_through_to_the_next():
    responder = Responder()
    responder.add("quiet", lambda turn: None)
    responder.add("answer", lambda turn: turn.say("here"))

    turn = Turn.from_request(request([{"role": "system", "content": "p"}]))
    assert responder.decide(turn) == Say("here")


def test_with_nothing_to_say_the_engine_stays_silent():
    # There is no way for an agent to hang up, so silence is how a call ends.
    responder = Responder()
    turn = Turn.from_request(request([{"role": "system", "content": "p"}]))
    assert isinstance(responder.decide(turn), Silence)


# --------------------------------------------------------------------------- answering


def parse_stream(chunks):
    events = []
    for chunk in chunks:
        if not chunk.startswith("data: "):
            continue
        payload = chunk[6:].strip()
        if payload == "[DONE]":
            events.append(("done", None))
            continue
        events.append(("chunk", json.loads(payload)))
    return events


def test_words_are_streamed_so_speech_starts_before_the_sentence_ends():
    turn = Turn.from_request(request([{"role": "system", "content": "p"}]))
    events = parse_stream(sse(turn, Say("two words")))

    contents = [
        event["choices"][0]["delta"]["content"]
        for kind, event in events
        if kind == "chunk" and event["choices"] and event["choices"][0]["delta"].get("content")
    ]
    assert "".join(contents).strip() == "two words"
    assert events[-1] == ("done", None)
    # The platform asks for usage, so a final usage-only chunk is sent.
    assert any(kind == "chunk" and event.get("usage") for kind, event in events)


def test_a_tool_call_is_streamed_in_the_shape_the_platform_executes():
    turn = Turn.from_request(request([{"role": "system", "content": "p"}]))
    events = parse_stream(sse(turn, Call("do_thing", id="7")))

    first = events[0][1]["choices"][0]["delta"]["tool_calls"][0]
    assert first["type"] == "function"
    assert first["function"]["name"] == "do_thing"
    assert json.loads(first["function"]["arguments"]) == {"id": "7"}
    assert events[1][1]["choices"][0]["finish_reason"] == "tool_calls"


def test_the_same_action_is_available_as_a_plain_body():
    turn = Turn.from_request(request([{"role": "system", "content": "p"}], stream=False))
    body = completion(turn, Say("hello"))
    assert body["choices"][0]["message"]["content"] == "hello"
    assert body["choices"][0]["finish_reason"] == "stop"

    called = completion(turn, Call("do_thing", id="7"))
    assert called["choices"][0]["finish_reason"] == "tool_calls"


def test_respond_hands_back_something_sendable_and_readable():
    responder = Responder()
    responder.add("only", lambda turn: turn.say("hello"))

    reply = responder.respond(request([{"role": "system", "content": "p"}]))
    assert isinstance(reply, Reply)
    assert reply.spoken == "hello"
    assert reply.tool is None
    assert reply.media_type == "text/event-stream"
    assert reply.json()["choices"][0]["message"]["content"] == "hello"
    assert "[only] say" in str(reply)
    assert list(reply.stream())[-1].startswith("data: [DONE]")


# --------------------------------------------------------------------------- odds and ends


@pytest.mark.parametrize(
    "said, expected",
    [
        ("it's four four seven one", "4471"),
        ("4 4 7 1", "4471"),
        ("forty one eleven", "4111"),
        ("4 1 double 1", "4111"),
        ("oh seven nine double five", "07955"),
        ("no digits here", ""),
    ],
)
def test_digits_read_out_however_they_were_said(said, expected):
    assert digits_said(said) == expected


def test_a_memo_survives_a_turn_the_transcript_forgot():
    # A caller talking over a scripted line leaves that turn interrupted, and
    # it does not come back in the messages, so anything that must be said once
    # needs a note of its own.
    memo = Memo()
    assert not memo.has("call-1", "notice")
    memo.note("call-1", "notice")
    assert memo.has("call-1", "notice")
    assert not memo.has("call-2", "notice")
    memo.forget("call-1")
    assert not memo.has("call-1", "notice")


async def test_the_tool_runner_separates_a_bad_name_from_bad_arguments():
    from assemblyai_agents import tool

    @tool
    async def do_thing(id: str) -> dict:
        """Do the thing.

        Args:
            id: which thing
        """
        return {"did": id}

    run = tool_runner([do_thing])
    assert await run("do_thing", {"id": "7"}) == {"did": "7"}
    assert run.names == ("do_thing",)

    with pytest.raises(LookupError):
        await run("nope", {})
    with pytest.raises(ValueError):
        await run("do_thing", {"wrong": "arg"})
