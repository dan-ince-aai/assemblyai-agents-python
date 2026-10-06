from dataclasses import dataclass
from typing import Any, Literal, Optional

from ._exceptions import ConfigurationError

ENCODINGS = ("audio/pcm", "audio/pcmu", "audio/pcma")
TRANSCRIPTION_MODES = ("balanced", "min_latency", "max_accuracy")
VOICE_FOCUS = ("near-field", "far-field")

# The agent turn classes a caller may not barge in on. These are the platform's
# own names for the turns, not a vocabulary invented here, and the set is closed
# — anything else is refused rather than left silently unprotected.
UNINTERRUPTIBLE_TURNS = ("greeting", "dtmf_prompt", "tool_refused_ask")

TURN_DETECTION_KEY = "turn_detection"
UNINTERRUPTIBLE_TURNS_KEY = "uninterruptible_turns"

MAX_KEYTERMS = 100
MAX_TRANSCRIPTION_PROMPT = 1750

Encoding = Literal["audio/pcm", "audio/pcmu", "audio/pcma"]
TranscriptionMode = Literal["balanced", "min_latency", "max_accuracy"]
VoiceFocus = Literal["near-field", "far-field"]


@dataclass(frozen=True, kw_only=True)
class AudioFormat:
    """Wire format of one audio stream.

    ``sample_rate`` exists only on the ``audio/pcm`` arm of the format union.
    The other two arms have no such field and none of those structs forbids unknown fields, so a
    ``sample_rate`` sent alongside a telephony encoding would be dropped in
    silence; it is refused here instead.
    """

    encoding: Encoding = "audio/pcm"
    sample_rate: Optional[Literal[24000]] = None

    def __post_init__(self) -> None:
        _reject_outside("format.encoding", self.encoding, ENCODINGS)
        if self.sample_rate is None:
            return
        if self.sample_rate != 24000:
            raise ConfigurationError(
                f"format.sample_rate={self.sample_rate!r} is not accepted: the only "
                f"rate the audio format carries is 24000."
            )
        if self.encoding != "audio/pcm":
            raise ConfigurationError(
                f"format.sample_rate is only carried by `audio/pcm`, and this format "
                f"is `{self.encoding}`. A rate sent with a telephony encoding is "
                f"dropped without a word, because the struct has no such field."
            )

    def to_dict(self) -> dict[str, Any]:
        emitted: dict[str, Any] = {"encoding": self.encoding}
        if self.sample_rate is not None:
            emitted["sample_rate"] = self.sample_rate
        return emitted


@dataclass(frozen=True, kw_only=True)
class AudioInput:
    """The agent's input configuration, as the stored row carries it.

    Two things about this helper are load-bearing.

    **It always emits ``type``.** The telephony path decodes the stored input as
    a union tagged on ``type``, so an untagged config fails to decode. The
    fallback then substitutes *nothing* — not the 1000/3000 ms the API documents
    — which leaves the platform's own session defaults in charge and discards
    the endpointing you configured, with only a server-side log line to show it.

    **Turn detection is deliberately not modelled.** Endpointing stays valid on
    the REST API and gets no typed field here yet, so pass it through ``extra``::

        AudioInput(extra={"turn_detection": {"min_silence": 400}})

    ``extra`` is merged into the emitted dict and refuses any key this class
    already models, so it can only add what is missing rather than quietly
    contradict a typed field.

    **One turn-detection key is modelled: ``uninterruptible_turns``.** It names
    the agent turns a caller may not cut in on, by turn class::

        AudioInput(uninterruptible_turns=["greeting"])

    Protection is a property of a turn, not a window on one: naming a class
    holds every turn of that class for its whole length. The names are a closed
    set — ``greeting``, ``dtmf_prompt`` and ``tool_refused_ask`` — so a typo is
    refused here rather than leaving a turn silently unprotected. The key is
    emitted inside the ``turn_detection`` block, where the server reads it, and
    joins whatever ``extra`` already puts there. ``[]`` is a real statement
    ("protect nothing") and is sent; unset sends nothing at all.

    **Two kinds of turn cannot be named, by construction.** A completion from
    your own model (``llm=``) is indistinguishable from any other model reply by
    the time it reaches the platform, so no static configuration can single it
    out — if you need a line said without interruption, put it in the
    ``greeting``. The ordinary reply to a caller's turn *is* distinguishable,
    but is not offered, because there is no way to say which replies to protect:
    naming the class would hold every reply in the call.

    **While a turn is protected the caller is not heard.** Speech arriving
    during one is discarded rather than queued, so the caller talks, is talked
    over, and what they said is never answered and never joins the
    conversation. Protect the turns that must land, not every turn.

    ``dtmf_prompt`` and ``tool_refused_ask`` carry one more caveat the greeting
    cannot hit: a protected turn waiting behind another one still plays after a
    barge-in cuts the turn in front of it, so the caller hears a fragment and
    then the protected line.
    """

    format: Optional[AudioFormat] = None
    keyterms: Optional[list[str]] = None
    transcription_mode: Optional[TranscriptionMode] = None
    continuous_partials: Optional[bool] = None
    transcription_prompt: Optional[str] = None
    language_codes: Optional[list[str]] = None
    voice_focus: Optional[VoiceFocus] = None
    voice_focus_threshold: Optional[float] = None
    uninterruptible_turns: Optional[list[str]] = None
    extra: Optional[dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.keyterms is not None and len(self.keyterms) > MAX_KEYTERMS:
            raise ConfigurationError(
                f"input.keyterms holds {len(self.keyterms)} terms; the API accepts at "
                f"most {MAX_KEYTERMS}."
            )
        if (
            self.transcription_prompt is not None
            and len(self.transcription_prompt) > MAX_TRANSCRIPTION_PROMPT
        ):
            raise ConfigurationError(
                f"input.transcription_prompt is {len(self.transcription_prompt)} "
                f"characters; the API accepts at most {MAX_TRANSCRIPTION_PROMPT}."
            )
        _reject_outside(
            "input.transcription_mode", self.transcription_mode, TRANSCRIPTION_MODES
        )
        _reject_outside("input.voice_focus", self.voice_focus, VOICE_FOCUS)
        _reject_out_of_range(
            "input.voice_focus_threshold", self.voice_focus_threshold, 0.0, 1.0
        )
        _reject_modelled_keys(self, self.extra)
        _reject_unknown_turns(self.uninterruptible_turns)
        _reject_turn_detection_collision(self.uninterruptible_turns, self.extra)

    def to_dict(self) -> dict[str, Any]:
        emitted: dict[str, Any] = {"type": "audio"}
        if self.format is not None:
            emitted["format"] = self.format.to_dict()
        for name in (
            "keyterms",
            "transcription_mode",
            "continuous_partials",
            "transcription_prompt",
            "language_codes",
            "voice_focus",
            "voice_focus_threshold",
        ):
            value = getattr(self, name)
            if value is not None:
                emitted[name] = value
        emitted.update(self.extra or {})
        if self.uninterruptible_turns is not None:
            # Nested rather than top level, because that is where the server
            # reads it. Added after `extra` so it joins an endpointing block
            # already there rather than replacing it.
            turn_detection = dict(emitted.get(TURN_DETECTION_KEY) or {})
            turn_detection[UNINTERRUPTIBLE_TURNS_KEY] = list(self.uninterruptible_turns)
            emitted[TURN_DETECTION_KEY] = turn_detection
        return emitted


@dataclass(frozen=True, kw_only=True)
class AudioOutput:
    """The agent's output configuration, as the stored row carries it.

    Always emits ``type``, for the reason given on :class:`AudioInput`.

    ``output.voice`` is deliberately not modelled: on the stored-agent path it is
    overwritten at bootstrap from the top-level ``voice``, and ``voice`` is
    required on create, so a value set here would never survive. Set the agent's
    ``voice`` instead.

    ``extra`` is merged into the emitted dict and refuses any key this class
    already models.
    """

    format: Optional[AudioFormat] = None
    volume: Optional[float] = None
    extra: Optional[dict[str, Any]] = None

    def __post_init__(self) -> None:
        _reject_out_of_range("output.volume", self.volume, 0.0, 100.0)
        _reject_modelled_keys(self, self.extra)

    def to_dict(self) -> dict[str, Any]:
        emitted: dict[str, Any] = {"type": "audio"}
        if self.format is not None:
            emitted["format"] = self.format.to_dict()
        if self.volume is not None:
            emitted["volume"] = self.volume
        emitted.update(self.extra or {})
        return emitted


def _nested_turn_detection(extra: Optional[dict]) -> dict:
    nested = (extra or {}).get(TURN_DETECTION_KEY)
    return nested if isinstance(nested, dict) else {}


def _reject_unknown_turns(turns: Optional[list]) -> None:
    if turns is None:
        return
    listed = ", ".join(f"`{name}`" for name in UNINTERRUPTIBLE_TURNS)
    if isinstance(turns, str) or not isinstance(turns, (list, tuple)):
        raise ConfigurationError(
            f"input.uninterruptible_turns={turns!r} is not a list. It names the "
            f"turn classes a caller may not cut in on, drawn from {listed}."
        )
    unknown = [name for name in turns if name not in UNINTERRUPTIBLE_TURNS]
    if not unknown:
        return
    named = ", ".join(repr(name) for name in unknown)
    raise ConfigurationError(
        f"input.uninterruptible_turns names {named}, which the API does not "
        f"know. The turn classes are {listed}. An unknown name is refused here "
        f"because the alternative is a turn that is silently never protected."
    )


def _reject_turn_detection_collision(
    turns: Optional[list], extra: Optional[dict]
) -> None:
    if turns is None or UNINTERRUPTIBLE_TURNS_KEY not in _nested_turn_detection(extra):
        return
    raise ConfigurationError(
        f"input.uninterruptible_turns is set and input.extra[{TURN_DETECTION_KEY!r}] "
        f"sets `{UNINTERRUPTIBLE_TURNS_KEY}` as well. That is one key written "
        f"twice, and the typed field wins silently. Keep one of them."
    )


def _reject_outside(field: str, value: Optional[str], allowed: tuple) -> None:
    if value is None or value in allowed:
        return
    listed = ", ".join(f"`{item}`" for item in allowed)
    raise ConfigurationError(f"{field}={value!r} is not one of {listed}.")


def _reject_out_of_range(
    field: str, value: Optional[float], low: float, high: float
) -> None:
    if value is None or low <= value <= high:
        return
    raise ConfigurationError(f"{field}={value!r} is outside {low}-{high}.")


def _reject_modelled_keys(config: Any, extra: Optional[dict]) -> None:
    # `extra` exists for keys this class does not model. A key it *does* model
    # would be two statements of the same thing, one of which wins silently.
    if not extra:
        return
    modelled = {"type", "extra"} | {
        field.name for field in config.__dataclass_fields__.values()
    }
    collisions = sorted(set(extra) & modelled)
    if not collisions:
        return
    listed = ", ".join(f"`{key}`" for key in collisions)
    raise ConfigurationError(
        f"{type(config).__name__}.extra sets {listed}, which this class already "
        f"models. `extra` is for keys it does not model — set the typed field "
        f"instead."
    )
