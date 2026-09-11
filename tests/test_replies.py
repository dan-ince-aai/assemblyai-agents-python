"""The bring-your-own-reply-engine surface.

Every case here is a shape the platform actually sends, or a trap it sets. The
comments say which, because that is the part a reader cannot guess.
"""

import json

import pytest
from assemblyai_agents.replies import (
    Call,
    Say,
    Silence,
    Turn,
    call_tool,
    digits_said,
    established,
    json_body,
    say,
    silence,
    stream,
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
    answer = call_tool("verify", caller_said="it's Maria", reference="", note=None, count=0)
    assert answer.arguments == {"caller_said": "it's Maria", "count": 0}
    assert established(a="x", b="", c=None) == {"a": "x"}


def test_the_answers_are_three_plain_functions():
    assert say("hello") == Say("hello")
    assert call_tool("do_thing", id="7") == Call("do_thing", {"id": "7"})
    assert silence() == Silence()


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
    events = parse_stream(stream(turn, say("two words")))

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
    events = parse_stream(stream(turn, call_tool("do_thing", id="7")))

    first = events[0][1]["choices"][0]["delta"]["tool_calls"][0]
    assert first["type"] == "function"
    assert first["function"]["name"] == "do_thing"
    assert json.loads(first["function"]["arguments"]) == {"id": "7"}
    assert events[1][1]["choices"][0]["finish_reason"] == "tool_calls"


def test_the_same_action_is_available_as_a_plain_body():
    turn = Turn.from_request(request([{"role": "system", "content": "p"}], stream=False))
    body = json_body(turn, say("hello"))
    assert body["choices"][0]["message"]["content"] == "hello"
    assert body["choices"][0]["finish_reason"] == "stop"

    called = json_body(turn, call_tool("do_thing", id="7"))
    assert called["choices"][0]["finish_reason"] == "tool_calls"