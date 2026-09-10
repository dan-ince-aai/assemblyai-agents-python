"""The system of record, mocked.

This is the only file that pretends. Swap each function for a call into your
scheduling system, your CRM, your payment gateway; the declaration, the reply
engine and the tests above it do not change.
"""

import re
import uuid
from datetime import date, datetime, timedelta

PRACTICE = "Fairview Dental"

# Patients, keyed by the reference printed on their reminder.
PATIENTS: dict[str, dict] = {
    "4471": {
        "reference": "4471",
        "first_name": "Maria",
        "last_name": "Delgado",
        "phone_number": "+14695550142",
        "last_seen": "2026-03-14",
        "balance_due": 0.0,
    },
    "8820": {
        "reference": "8820",
        "first_name": "Tomas",
        "last_name": "Okonkwo",
        "phone_number": "+19725550461",
        "last_seen": "2025-09-02",
        "balance_due": 45.0,
    },
}

APPOINTMENTS: list[dict] = []
CALLBACKS: list[dict] = []
_IN_FLIGHT: dict | None = None

# Slots the practice has free, as (date, time) pairs.
_TIMES = ("09:00", "09:30", "11:00", "14:00", "15:30", "16:00")


from assemblyai_agents.byo import digits_said


def _digits(value: str) -> str:
    """Digits from figures or from words, so a spoken reference resolves."""
    return digits_said(value) if re.search(r"[a-z]", (value or "").lower()) else re.sub(r"\D", "", value or "")


def _normalise(value: str) -> str:
    return re.sub(r"[^a-z ]", "", (value or "").lower()).strip()


def find_by_reference(reference: str) -> dict | None:
    return PATIENTS.get(_digits(reference)[-4:])


def find_by_phone(phone_number: str) -> dict | None:
    wanted = _digits(phone_number)[-10:]
    if not wanted:
        return None
    return next(
        (p for p in PATIENTS.values() if _digits(p["phone_number"]).endswith(wanted)), None
    )


def remember_in_flight(patient: dict | None) -> None:
    """Hold the patient a pre-connect lookup matched, for the rest of the call.

    Process-local, which is enough for one worker and a demo. Anything real
    would key this by call id, or fetch it again in each tool.
    """
    global _IN_FLIGHT
    _IN_FLIGHT = patient


def resolve(reference: str | None) -> dict | None:
    return find_by_reference(reference) if reference else _IN_FLIGHT


def extract_name(caller_said: str) -> str:
    """The name out of whatever the caller said.

    Extraction happens here, not in the reply engine, because the platform
    refuses a tool argument whose value was never spoken: the caller's own
    words go over the wire and the reading is done at this end.
    """
    text = (caller_said or "").strip()
    if text.endswith("?") or re.match(r"(?i)^(what|why|who|how|when|where|can|did|do|is|are)\b", text):
        return ""
    cleaned = re.sub(
        r"(?i)\b(uh|um|er|yeah|yes|no|hi|hello|this|is|it|its|it's|i|i'm|am|my|name|"
        r"speaking|sure|that's|mr|mrs|ms|miss|dr|the|of|course|told|you|just|said|and)\b",
        " ",
        re.sub(r"[^A-Za-z' -]", " ", text),
    )
    words = [word for word in cleaned.split() if len(word) > 1]
    return " ".join(words[:3])


def name_matches(patient: dict, said: str) -> bool:
    """Forgiving about middle names and case, strict about the surname."""
    heard = _normalise(said).split()
    return bool(heard) and _normalise(patient["last_name"]) in heard


def open_slots(from_date: str | None = None, limit: int = 6) -> list[dict]:
    """Free appointments, soonest first. Weekends are closed."""
    start = date.fromisoformat(from_date) if from_date else date.today() + timedelta(days=1)
    taken = {(a["date"], a["time"]) for a in APPOINTMENTS}
    found: list[dict] = []
    day = start
    while len(found) < limit and (day - start).days < 21:
        if day.weekday() < 5:
            for time_of_day in _TIMES:
                if (day.isoformat(), time_of_day) not in taken:
                    found.append({"date": day.isoformat(), "time": time_of_day})
                if len(found) >= limit:
                    break
        day += timedelta(days=1)
    return found


def book(patient: dict, slot_date: str, slot_time: str, reason: str = "check-up") -> dict:
    if any(a["date"] == slot_date and a["time"] == slot_time for a in APPOINTMENTS):
        return {"booked": False, "problem": "slot_taken"}
    record = {
        "confirmation": uuid.uuid4().hex[:6].upper(),
        "reference": patient["reference"],
        "date": slot_date,
        "time": slot_time,
        "reason": reason,
        "booked_at": datetime.now().isoformat(timespec="seconds"),
    }
    APPOINTMENTS.append(record)
    return {"booked": True, **record}


def request_callback(patient: dict | None, reason: str, note: str = "") -> dict:
    record = {
        "reference": (patient or {}).get("reference"),
        "reason": reason,
        "note": note,
        "at": datetime.now().isoformat(timespec="seconds"),
    }
    CALLBACKS.append(record)
    return {"logged": True, **record}


def spoken_date(value: str) -> str:
    """An ISO date as a person would say it: "Tuesday the 15th of September"."""
    when = date.fromisoformat(value)
    day = when.day
    suffix = "th" if 11 <= day <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    return f"{when:%A} the {day}{suffix} of {when:%B}"


def spoken_time(value: str) -> str:
    hour, minute = (int(part) for part in value.split(":"))
    suffix = "in the morning" if hour < 12 else "in the afternoon"
    twelve = hour % 12 or 12
    return (f"{twelve} o'clock" if minute == 0 else f"{twelve} {minute:02d}") + f" {suffix}"


def reset() -> None:
    global _IN_FLIGHT
    _IN_FLIGHT = None
    APPOINTMENTS.clear()
    CALLBACKS.clear()
