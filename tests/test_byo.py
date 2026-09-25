"""The bring-your-own-reply-engine surface.

Every case here is a shape the platform actually sends, or a trap it sets. The
comments say which, because that is the part a reader cannot guess.
"""

import json

import pytest
from assemblyai_agents.byo import (
    Call,
    Say,
    Silence,
    Turn,
    call_tool,
    digits_said,
    json_body,
    say,
    silence,
    stream,
)

# The guidance the platform appends to every failed result, as it sends it.
FAILED_GUIDANCE = (
    "\n[The tool did not run, so nothing has changed. You called it with: "
    '{"id": "7"}.\nFind a correct value before you reply:\n'
    "- reuse a value exactly as an earlier tool result returned it;\n"
    "Only ask the caller if none of those has it. Never invent a value.]"
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
    # caller.
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


def test_a_json_error_body_from_a_failed_tool_is_not_a_result():
    # Your endpoint's error body reaches the transcript as it was sent, so it
    # parses as JSON; only the guidance the platform appends says it failed.
    turn = Turn.from_request(
        request([
            {"role": "system", "content": "prompt"},
            tool_call("c1", "do_thing", {"id": "7"}),
            {"role": "tool", "tool_call_id": "c1", "content": json.dumps({"detail": "unauthorized"}) + FAILED_GUIDANCE},
        ])
    )
    assert turn.pending.ran is False
    assert turn.pending.note == '{"detail": "unauthorized"}'
    assert turn.result_of("do_thing") is None


def test_a_result_delivered_late_still_ran():
    # The platform appends this note to a result the caller had moved on from.
    late = (
        json.dumps({"value": 42})
        + "\n\n[This result arrived after the caller had already moved on, so it was "
        "never spoken to them. Incorporate it into your next reply if it is still "
        "relevant; otherwise ignore it.]"
    )
    turn = Turn.from_request(
        request([
            {"role": "system", "content": "prompt"},
            tool_call("c1", "do_thing", {"id": "7"}),
            {"role": "tool", "tool_call_id": "c1", "content": late},
        ])
    )
    assert turn.pending.ran is True
    assert turn.pending.get("value") == 42


def test_a_result_cut_at_the_size_cap_ran_but_has_no_value():
    turn = Turn.from_request(
        request([
            {"role": "system", "content": "prompt"},
            tool_call("c1", "do_thing", {"id": "7"}),
            {"role": "tool", "tool_call_id": "c1", "content": '{"rows": [1, 2\n…[truncated]'},
        ])
    )
    assert turn.pending.ran is True
    assert turn.pending.value is None
    assert turn.pending.note == "truncated"


def test_the_platforms_duplicate_call_answer_is_not_a_result():
    turn = Turn.from_request(
        request([
            {"role": "system", "content": "prompt"},
            tool_call("c1", "do_thing", {"id": "7"}),
            {
                "role": "tool",
                "tool_call_id": "c1",
                "content": "This function is already executing with these arguments. "
                "Result is pending from the previous call.",
            },
        ])
    )
    assert turn.pending.ran is False


def test_words_spoken_with_a_call_do_not_answer_its_result():
    # The words and the call arrive folded into one assistant message, which
    # sits before the result.
    turn = Turn.from_request(
        request([
            {"role": "system", "content": "prompt"},
            {**tool_call("c1", "do_thing", {"id": "7"}), "content": "One moment."},
            {"role": "tool", "tool_call_id": "c1", "content": json.dumps({"ok": True})},
        ])
    )
    assert turn.spoken == ("One moment.",)
    assert turn.pending is not None and turn.pending.get("ok") is True


def test_a_user_message_wins_over_a_quoted_argument_in_a_platform_note():
    # The completion note writes arguments out as Python values, which puts
    # double quotes around a string holding an apostrophe.
    turn = Turn.from_request(
        request([
            {"role": "system", "content": "prompt"},
            {"role": "user", "content": "my name is Maria"},
            {"role": "system", "content": 'The function call verify(name="it\'s Maria") has just completed.'},
        ])
    )
    assert turn.caller_said == "my name is Maria"


# --------------------------------------------------------------------------- deciding


def test_a_tool_call_keeps_every_argument_as_given():
    # The platform does not check a customer LLM's arguments, not even for
    # required ones, so a dropped argument would reach the tool as a missing
    # keyword rather than be caught.
    answer = call_tool("verify", caller_said="it's Maria", reference="", note=None, count=0)
    assert answer.arguments == {"caller_said": "it's Maria", "reference": "", "note": None, "count": 0}


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

# --------------------------------------------------------------------------- speaking over a tool


def _turn():
    return Turn.from_request(request([{"role": "user", "content": "what is my balance"}]))


def _deltas(frames):
    """Every (delta, finish_reason) in the response. The usage frame carries no
    choices, so it is skipped rather than indexed into."""
    out = []
    for frame in frames:
        payload = frame[len("data: ") :].strip()
        if payload == "[DONE]":
            continue
        choices = json.loads(payload).get("choices") or []
        if not choices:
            continue
        out.append((choices[0].get("delta") or {}, choices[0].get("finish_reason")))
    return out


def test_a_bare_call_still_says_nothing():
    """The default is unchanged: a tool call on its own speaks no words."""
    frames = list(stream(_turn(), call_tool("check_balance", account="1")))

    assert not any(d.get("content") for d, _ in _deltas(frames))


def test_a_call_and_its_words_travel_in_one_response():
    """The platform acts on the call only once the response finishes, so the
    order within it is cosmetic; the call is written first."""
    frames = list(stream(_turn(), call_tool("check_balance", saying="One moment.", account="1")))
    deltas = _deltas(frames)

    assert deltas[0][0].get("tool_calls")
    assert "".join(d.get("content") or "" for d, _ in deltas).strip() == "One moment."


def test_the_response_still_finishes_as_a_tool_call():
    """The words are carried by a response whose reason is `tool_calls`, the
    finish an OpenAI-compatible client expects after a tool call."""
    frames = list(stream(_turn(), call_tool("check_balance", saying="One moment.", account="1")))

    assert [r for _, r in _deltas(frames) if r] == ["tool_calls"]


def test_saying_is_the_spoken_line_not_a_tool_argument():
    """The one shape this changes: `saying` is keyword-only on `call_tool`, so a
    tool whose own argument is named `saying` no longer receives it. Build the
    `Call` directly if you need that."""
    assert call_tool("t", saying="One moment.", account="1").arguments == {"account": "1"}
    assert Call("t", {"saying": "a value"}).saying is None


@pytest.mark.parametrize(
    ("spoken", "digits"),
    [
        ("it's four four seven one", "4471"),
        ("forty one eleven", "4111"),
        ("4 1 double 1", "4111"),
        ("oh seven triple nine", "07999"),
        ("twenty", "20"),
        ("thirty oh", "300"),
        ("no digits here", ""),
    ],
)
def test_digits_are_read_out_of_speech(spoken, digits):
    assert digits_said(spoken) == digits
