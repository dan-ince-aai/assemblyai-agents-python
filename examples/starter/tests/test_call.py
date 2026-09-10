"""What the agent does, driven through the same loop the platform uses.

These are the tests worth having on a voice agent: whole calls, asserted on
what was said and what ended up in the system of record. Each one runs offline
in milliseconds, so they belong in CI.
"""

import rehearse
import reply
import store


def agent_lines(transcript) -> list[str]:
    return [text for speaker, text in transcript if speaker == "agent"]


def spoken(transcript) -> str:
    return " ".join(agent_lines(transcript))


def test_the_happy_path_books_an_appointment_and_confirms_it():
    transcript = rehearse.run(rehearse.HAPPY, quiet=True)

    assert len(store.APPOINTMENTS) == 1
    assert "You are booked in for" in spoken(transcript)
    assert store.APPOINTMENTS[0]["confirmation"] in spoken(transcript).replace(" ", "")


def test_nothing_from_the_record_is_said_before_the_name_is_checked():
    # The reference alone is not identification, so the notice, the balance and
    # anything else on the record stay unsaid until verify_caller passes.
    transcript = rehearse.run(["It's four four seven one."], quiet=True)

    said = spoken(transcript)
    assert reply.NOTICE not in said
    assert "outstanding" not in said
    assert "Maria" not in said


def test_the_notice_is_said_once_even_if_the_caller_talks_over_it():
    # An interrupted turn does not always come back in the transcript, so the
    # engine keeps its own note. Without that the notice repeats forever.
    transcript = rehearse.run(rehearse.HAPPY, quiet=True)
    assert spoken(transcript).count(reply.NOTICE) == 1

    turn_count = sum(1 for speaker, _ in transcript if speaker == "agent")
    assert turn_count >= 4


def test_a_wrong_name_is_named_back_and_a_question_does_not_cost_an_attempt():
    transcript = rehearse.run(rehearse.WRONG_NAME, quiet=True)

    said = spoken(transcript)
    assert "does not match the record" in said
    # The question in the middle of that scenario is answered, not scored.
    assert "I need the name on the record first" in said
    # Three real attempts, then a person picks it up.
    assert store.CALLBACKS and "could not verify" in store.CALLBACKS[0]["reason"]
    assert not store.APPOINTMENTS


def test_a_caller_matched_by_their_number_never_reads_out_a_reference():
    transcript = rehearse.run(rehearse.BY_PHONE, caller_number="+14695550142", quiet=True)

    assert "record from the number you are calling from" in agent_lines(transcript)[0]
    assert reply.REFERENCE_ASK not in spoken(transcript)
    assert len(store.APPOINTMENTS) == 1


def test_the_call_goes_quiet_once_it_is_finished():
    # An agent cannot hang up a phone call, so the end of a call is silence.
    # Anything else is a new question the caller did not ask for.
    transcript = rehearse.run(rehearse.HAPPY + ["Thanks, bye."], quiet=True)

    assert agent_lines(transcript)[-1] == "(silence)"


def test_an_objection_gets_the_practice_position_and_not_an_invention():
    # With the model off the written position goes out unchanged, which is the
    # behaviour to rely on when the model is slow or unavailable.
    transcript = rehearse.run(rehearse.OBJECTION, quiet=True)

    said = spoken(transcript)
    assert "nothing sooner in the diary" in said
    assert "call you back today" in said


def test_a_slot_already_taken_is_never_offered():
    first = store.open_slots(limit=1)[0]

    transcript = rehearse.run(
        [
            "It's four four seven one.",
            "Maria Delgado.",
            "Tomorrow morning please.",
            "The first one.",
        ],
        quiet=True,
        # Booked by someone else between the reset and the call.
        prepare=lambda: store.book(store.PATIENTS["8820"], first["date"], first["time"]),
    )

    assert len(store.APPOINTMENTS) == 2
    booked = [a for a in store.APPOINTMENTS if a["reference"] == "4471"]
    assert booked and (booked[0]["date"], booked[0]["time"]) != (first["date"], first["time"])
    assert "You are booked in for" in spoken(transcript)
