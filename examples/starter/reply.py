"""What the agent says next: the part you write.

The call is split into stages, and each stage owns a few turns and hands over
when its own test says it is done. That split is the point: the ends of a call
are usually a script, the middle is a conversation, and one handler trying to
be both turns into a tangle of conditions.

    identify   scripted   ask for the reference, check the name
    notice     scripted   the line that must be said, exactly, once
    book       mixed      collect a preference, offer slots, confirm, book
    close      scripted   confirm and go quiet

Scripted stages never call the model, so the wording cannot drift. The booking
stage uses it for the parts a script cannot cover: what the caller meant, and
answering an objection with a position that is settled in code.
"""

import re

from assemblyai_agents.byo import Turn, call_tool, digits_said, say, silence

import model
import store
from flow import Memo, Stages

# Lines that must go out as written.
NOTICE = (
    "Before we go on, this call is recorded, and anything you tell me is added to your "
    "patient record."
)
REFERENCE_ASK = "Could you read me the four digit reference on your reminder?"
NAME_ASK = "And could you give me your full name, so I can check the record?"
WHEN_ASK = "When would suit you to come in?"
CLOSING = f"Thanks for calling {store.PRACTICE}. Take care."

# What the practice will and will not do, decided here rather than by a model.
POSITIONS = {
    "no_slot_sooner": (
        "The next free appointments are the ones I read out; there is nothing sooner in the "
        "diary. If it is urgent, a member of the team can call you back today."
    ),
    "cost": (
        "I cannot quote for treatment on the phone, because it depends on what the dentist "
        "finds. Reception can go through costs when you come in."
    ),
    "other": (
        "I can book an appointment, or arrange for a member of the team to call you back. "
        "Anything clinical needs to be a person."
    ),
}
ASK_OPTIONS = ["book_appointment", "reschedule", "cancel", "cost", "clinical_question", "no_slot_sooner"]

# Anything that must be said exactly once is noted here, because an interrupted
# turn does not reliably come back in the transcript.
memo = Memo()

_YES = re.compile(
    r"(?i)\b(yes|yeah|yep|yup|sure|certainly|absolutely|correct|ok|okay|fine|please do|"
    r"go ahead|go on|that works|of course|perfect|great)\b|that'?s (right|fine)|uh[- ]?huh|mm?[- ]?hm+"
)
_NO = re.compile(r"(?i)\b(no|nope|nah|not|don'?t|can'?t|cannot|won'?t|never|rather not)\b|uh[- ]?uh")
_TIME_WORDS = re.compile(
    r"(?i)\b(morning|afternoon|evening|monday|tuesday|wednesday|thursday|friday|"
    r"next week|this week|tomorrow|today|\d{1,2}(am|pm|:\d\d))\b"
)
# Not times: a caller pushing back on what was offered.
_PUSHBACK = re.compile(r"(?i)\b(sooner|earlier|anything else|nothing else|urgent|killing me|too (late|far))\b")


def call_key(turn: Turn) -> str:
    """One call, for the memo. The reference once known, otherwise the caller."""
    return reference(turn) or "unidentified"


def reference(turn: Turn) -> str:
    """The patient reference, from the pre-connect lookup or from a tool result."""
    if turn.preconnect.get("reference"):
        return str(turn.preconnect["reference"])
    for name in ("verify_caller", "find_patient"):
        result = turn.result_of(name)
        if result and result.get("reference"):
            return str(result.get("reference"))
    return ""


def verified(turn: Turn) -> bool:
    result = turn.result_of("verify_caller")
    return bool(result and result.get("verified"))


def booked(turn: Turn):
    result = turn.result_of("book_appointment")
    return result.value if result and result.get("booked") else None


def agrees(said: str, question: str) -> bool:
    """Keywords first, then the model, because "uh-huh" is a yes."""
    if _YES.search(said or "") and not _NO.search(said or ""):
        return True
    if _NO.search(said or "") and not _YES.search(said or ""):
        return False
    return bool(model.agrees(said, question))


stages = Stages()


# --------------------------------------------------------------------------- identify


@stages.stage("identify", until=verified)
def identify(turn: Turn):
    """Find the record, then check the caller is the patient.

    Nothing on the record is discussed before this stage is done, which is why
    it is first and why find_patient returns nothing identifying.
    """
    pending = turn.pending
    if pending and pending.name == "find_patient":
        if not pending.get("found"):
            return say(f"I could not find a record for that reference. {REFERENCE_ASK}")
        return say(NAME_ASK)

    if pending and pending.name == "verify_caller":
        problem = pending.get("problem")
        # Three real attempts is enough. Counted off the transcript rather than
        # held anywhere, and a question does not count as an attempt because
        # the tool reports it as no_name_heard.
        attempts = sum(
            1
            for result in turn.results
            if result.name == "verify_caller" and result.get("problem") != "no_name_heard"
        )
        if problem == "no_match" and attempts >= 3:
            return call_tool(
                "request_callback",
                reason="could not verify the caller",
                note=f"three attempts, last heard {pending.get('name_heard')!r}",
            )
        if problem == "no_name_heard":
            # They asked something rather than answering. Nothing is implied
            # about their name, and no attempt is spent.
            return say(f"Sorry, I need the name on the record first. {NAME_ASK}")
        if problem == "no_record_in_context":
            return say(REFERENCE_ASK)
        heard = pending.get("name_heard")
        return say(
            (f"I heard {heard}, and that " if heard else "That ")
            + "does not match the record, so I cannot go into it. "
            + NAME_ASK
        )

    if pending and pending.name == "request_callback":
        memo.note(call_key(turn), "closed")
        return say(
            "I have asked a member of the team to call you back so they can check the "
            f"record with you properly. {CLOSING}"
        )

    said = turn.caller_said
    if not said:
        return say(REFERENCE_ASK)

    # A record in hand means the next thing needed is the name.
    if reference(turn) or turn.preconnect:
        return call_tool("verify_caller", caller_said=said, reference=reference(turn))

    # Callers read a reference out in words as often as in figures.
    if len(digits_said(said)) >= 4:
        return call_tool("find_patient", reference_said=said)
    return say(REFERENCE_ASK)


# --------------------------------------------------------------------------- the notice


@stages.stage("notice", until=lambda turn: memo.has(call_key(turn), "notice"))
def notice(turn: Turn):
    """The line that has to be said, as written, once.

    It goes out the moment identification succeeds and is noted immediately, so
    a caller talking over it does not make it repeat.
    """
    memo.note(call_key(turn), "notice")
    result = turn.result_of("verify_caller")
    first_name = (result.get("first_name") if result else "") or "there"
    balance = (result.get("balance_due") if result else 0) or 0
    owing = f" Our records show {balance:.2f} pounds outstanding on the account." if balance else ""
    return say(f"Thank you, {first_name}. {NOTICE}{owing} {WHEN_ASK}")


# --------------------------------------------------------------------------- book


def told_them(turn: Turn) -> bool:
    """The booking is done and the caller has been told.

    Not simply "a booking exists": a stage that stands down the moment its tool
    succeeds hands over before it has said so, and the caller never hears the
    confirmation.
    """
    key = call_key(turn)
    return memo.has(key, "booked") or memo.has(key, "closed")


@stages.stage("book", until=told_them)
def book(turn: Turn):
    """Offer what is free, confirm one, book it.

    The parts a script cannot do are handed to the model: what the caller
    meant, and answering an objection with a position from POSITIONS.
    """
    said = turn.caller_said
    pending = turn.pending

    if pending and pending.name == "find_appointments":
        slots = pending.get("slots") or []
        if not slots:
            return say("I have nothing free in the next three weeks. Shall I have someone call you?")
        offered = "; or ".join(slot["spoken"] for slot in slots[:2])
        return say(f"I can offer {offered}. Would either of those work?")

    if pending and pending.name == "book_appointment":
        if pending.get("booked"):
            memo.note(call_key(turn), "booked")
            return say(
                f"You are booked in for {pending.get('spoken')}. Your confirmation is "
                f"{' '.join(str(pending.get('confirmation')))}. {CLOSING}"
            )
        if pending.get("problem") == "slot_taken":
            return call_tool("find_appointments")
        return say("I could not book that. Shall I have someone from the team call you back?")

    if pending and pending.name == "request_callback":
        memo.note(call_key(turn), "closed")
        return say(f"That is with the team and they will call you. {CLOSING}")



    if not said:
        return say(WHEN_ASK)

    # Did they pick one of the slots that were offered?
    offered = _slots_offered(turn)
    if offered:
        chosen = _chosen_slot(said, offered)
        if chosen:
            return call_tool(
                "book_appointment",
                slot_date=chosen["date"],
                slot_time=chosen["time"],
                reference=reference(turn),
            )
        # Checked before the time words, because "nothing sooner?" is a
        # complaint about the diary and searching it again just reads the same
        # two slots back.
        if _PUSHBACK.search(said) or (_NO.search(said) and not _TIME_WORDS.search(said)):
            return _position(turn, "no_slot_sooner", said)
        if agrees(said, "Would either of those work?"):
            return call_tool(
                "book_appointment",
                slot_date=offered[0]["date"],
                slot_time=offered[0]["time"],
                reference=reference(turn),
            )

    if _TIME_WORDS.search(said) or (turn.said_before(WHEN_ASK) and not offered):
        return call_tool("find_appointments", preference_said=said)

    # Not a time, not a yes or no: work out what they want, and answer from a
    # position rather than improvising one.
    wanted = model.choose(said, ASK_OPTIONS)
    if wanted in ("book_appointment", "reschedule"):
        return call_tool("find_appointments", preference_said=said)
    if wanted == "clinical_question":
        return call_tool("request_callback", reason="clinical question", note=said, reference=reference(turn))
    if wanted in POSITIONS:
        return _position(turn, wanted, said)
    return _position(turn, "other", said)


def _position(turn: Turn, key: str, said: str):
    """Say a settled position, in words that answer what the caller said."""
    position = POSITIONS[key]
    return say(model.deliver(position, said) or f"{position} {WHEN_ASK}")


def _slots_offered(turn: Turn) -> list:
    result = turn.result_of("find_appointments")
    return list(result.get("slots") or []) if result else []


def _chosen_slot(said: str, offered: list) -> dict | None:
    """Which offered slot the caller picked, by day, time or "the first one"."""
    lowered = (said or "").lower()
    for index, word in enumerate(("first", "second", "third", "fourth")):
        if word in lowered and index < len(offered):
            return offered[index]
    for slot in offered:
        day, hour = slot["spoken"].split(" the ")[0].lower(), slot["time"][:2]
        if day in lowered and (hour.lstrip("0") in lowered or slot["time"] in lowered):
            return slot
    for slot in offered:
        if slot["spoken"].split(" the ")[0].lower() in lowered:
            return slot
    return None


# --------------------------------------------------------------------------- close


@stages.stage("close")
def close(turn: Turn):
    """The call is done. Say goodbye once, then stay quiet.

    There is no way for an agent to hang up a phone call, so silence is how a
    finished call ends. Anything else is the agent inventing a new question.
    """
    key = call_key(turn)
    # The stage that finished the job said goodbye as part of confirming it, so
    # there is nothing to add. Saying it twice, or asking a fresh question, is
    # how the end of a call turns into a loop.
    if memo.has(key, "closed") or memo.has(key, "booked"):
        return silence()
    memo.note(key, "closed")
    return say(CLOSING)


def decide(turn: Turn):
    return stages.decide(turn) or silence()
