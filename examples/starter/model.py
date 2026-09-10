"""A model for the turns a script cannot cover, and nothing else.

Two calls are worth making. One classifies what the caller wants when keywords
cannot. The other says a settled thing conversationally: the position is
decided in code and handed over as the thing to convey, so the model chooses
the words and not the policy.

Point MODEL_BASE_URL at any OpenAI-compatible endpoint. Unset it, or set
MODEL=off, and the reply engine falls back to its written wording, which is
always correct if less fluent.
"""

import os

import httpx

BASE_URL = os.environ.get("MODEL_BASE_URL", "").rstrip("/")
API_KEY = os.environ.get("MODEL_API_KEY", "")
NAME = os.environ.get("MODEL_NAME", "")
# The platform gives a reply endpoint about ten seconds, and the caller hears
# silence while it waits, so this is deliberately short.
TIMEOUT = float(os.environ.get("MODEL_TIMEOUT", "5"))
ENABLED = bool(BASE_URL and API_KEY and NAME) and os.environ.get("MODEL", "on") != "off"

VOICE_RULES = (
    "You are a receptionist on a recorded phone call. Reply in one or two short spoken "
    "sentences. No formatting, no lists, no emoji. Do not invent policy, prices, dates or "
    "availability. Never mention being an AI."
)


def _complete(system: str, user: str, max_tokens: int = 200) -> str | None:
    if not ENABLED:
        return None
    try:
        response = httpx.post(
            f"{BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
            json={
                "model": NAME,
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
            timeout=TIMEOUT,
        )
        if response.status_code != 200:
            print(f"[model] {response.status_code}: {response.text[:120]}", flush=True)
            return None
        return ((response.json()["choices"][0]["message"].get("content")) or "").strip() or None
    except Exception as exc:
        # A slow or broken model must never take a call down.
        print(f"[model] {type(exc).__name__}: {exc}", flush=True)
        return None


def choose(said: str, options: list[str]) -> str | None:
    """Which of these the caller is asking for, or None."""
    answer = _complete(
        "Classify what a caller wants. Answer with one label from the list and nothing else. "
        "Use 'other' if none fit.",
        f"Labels: {', '.join(options + ['other'])}\n\nCaller said: {said!r}",
        max_tokens=20,
    )
    if not answer:
        return None
    label = answer.strip().strip(".'\"").lower().replace(" ", "_")
    return label if label in options else None


def agrees(said: str, question: str) -> bool | None:
    """Whether a reply means yes, when the words alone do not settle it.

    "Uh-huh", "go on then", "that's fine by me" are all yes and none of them is
    the word yes. Getting this wrong on a confirmation is how a call loops.
    """
    answer = _complete(
        "Decide whether a reply to a yes or no question means yes, no, or neither. "
        "Answer with one word: yes, no, or unclear.",
        f"Question: {question!r}\nReply: {said!r}",
        max_tokens=8,
    )
    if not answer:
        return None
    word = answer.strip().strip(".'\"").lower()
    return True if word == "yes" else False if word == "no" else None


def deliver(position: str, said: str) -> str | None:
    """Say a settled position in a way that answers what the caller said."""
    return _complete(
        VOICE_RULES
        + " You are given the practice's position, which is settled and must be conveyed. "
        "Acknowledge what the caller said, convey that position, and end with a question that "
        "moves things forward. Do not add options that are not in the position.",
        f"Caller said: {said!r}\nPosition to convey: {position!r}",
        max_tokens=220,
    )


def status() -> dict:
    return {"enabled": ENABLED, "model": NAME if ENABLED else None}
