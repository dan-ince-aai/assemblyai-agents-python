import dataclasses

import pytest
from assemblyai_agents import AudioFormat, AudioInput, AudioOutput, ConfigurationError


def test_an_input_with_nothing_set_still_emits_the_type_tag():
    # Telephony decodes the stored input as a tagged union on `type`, and a
    # failed decode substitutes nothing at all — not the documented 1000/3000 ms
    # — so the customer's endpointing is discarded in silence.
    assert AudioInput().to_dict() == {"type": "audio"}


def test_an_output_with_nothing_set_still_emits_the_type_tag():
    assert AudioOutput().to_dict() == {"type": "audio"}


def test_a_fully_populated_input_emits_every_modelled_key():
    audio_input = AudioInput(
        format=AudioFormat(encoding="audio/pcm", sample_rate=24000),
        keyterms=["margherita", "calzone"],
        transcription_mode="max_accuracy",
        continuous_partials=True,
        transcription_prompt="The caller is ordering pizza.",
        language_codes=["en_us"],
        voice_focus="far-field",
        voice_focus_threshold=0.4,
    )

    assert audio_input.to_dict() == {
        "type": "audio",
        "format": {"encoding": "audio/pcm", "sample_rate": 24000},
        "keyterms": ["margherita", "calzone"],
        "transcription_mode": "max_accuracy",
        "continuous_partials": True,
        "transcription_prompt": "The caller is ordering pizza.",
        "language_codes": ["en_us"],
        "voice_focus": "far-field",
        "voice_focus_threshold": 0.4,
    }


def test_a_fully_populated_output_emits_every_modelled_key():
    audio_output = AudioOutput(format=AudioFormat(encoding="audio/pcmu"), volume=80.0)

    assert audio_output.to_dict() == {
        "type": "audio",
        "format": {"encoding": "audio/pcmu"},
        "volume": 80.0,
    }


def test_the_escape_hatch_carries_a_key_the_helper_does_not_model():
    # Endpointing is deliberately unmodelled and still valid on the API.
    audio_input = AudioInput(
        extra={"turn_detection": {"min_silence": 400, "max_silence": 2000}}
    )

    assert audio_input.to_dict() == {
        "type": "audio",
        "turn_detection": {"min_silence": 400, "max_silence": 2000},
    }


def test_the_escape_hatch_refuses_a_key_the_helper_already_models():
    with pytest.raises(ConfigurationError) as exc_info:
        AudioInput(keyterms=["margherita"], extra={"keyterms": ["calzone"]})

    assert "`keyterms`" in str(exc_info.value)


def test_the_escape_hatch_refuses_the_type_tag():
    with pytest.raises(ConfigurationError) as exc_info:
        AudioInput(extra={"type": "text"})

    assert "`type`" in str(exc_info.value)


def test_an_output_escape_hatch_key_survives():
    assert AudioOutput(extra={"voice": "ivy"}).to_dict() == {
        "type": "audio",
        "voice": "ivy",
    }


def test_more_than_a_hundred_keyterms_is_refused():
    with pytest.raises(ConfigurationError) as exc_info:
        AudioInput(keyterms=[f"term_{index}" for index in range(101)])

    assert "100" in str(exc_info.value)


def test_a_hundred_keyterms_is_accepted():
    audio_input = AudioInput(keyterms=[f"term_{index}" for index in range(100)])

    assert len(audio_input.to_dict()["keyterms"]) == 100


def test_an_over_long_transcription_prompt_is_refused():
    with pytest.raises(ConfigurationError) as exc_info:
        AudioInput(transcription_prompt="x" * 1751)

    assert "1750" in str(exc_info.value)


@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_a_voice_focus_threshold_outside_the_range_is_refused(threshold):
    with pytest.raises(ConfigurationError) as exc_info:
        AudioInput(voice_focus_threshold=threshold)

    assert "voice_focus_threshold" in str(exc_info.value)


@pytest.mark.parametrize("volume", [-1.0, 100.1])
def test_a_volume_outside_the_range_is_refused(volume):
    with pytest.raises(ConfigurationError) as exc_info:
        AudioOutput(volume=volume)

    assert "volume" in str(exc_info.value)


def test_an_unknown_transcription_mode_is_refused():
    with pytest.raises(ConfigurationError) as exc_info:
        AudioInput(transcription_mode="fastest")

    assert "transcription_mode" in str(exc_info.value)


def test_an_unknown_voice_focus_is_refused():
    with pytest.raises(ConfigurationError) as exc_info:
        AudioInput(voice_focus="mid-field")

    assert "voice_focus" in str(exc_info.value)


def test_an_unknown_encoding_is_refused():
    with pytest.raises(ConfigurationError) as exc_info:
        AudioFormat(encoding="audio/opus")

    assert "encoding" in str(exc_info.value)


def test_a_sample_rate_on_a_telephony_encoding_is_refused():
    # `sample_rate` only exists on the `audio/pcm` arm of the format union, and
    # none of those structs forbids unknown fields, so a stray one would be dropped without a word.
    with pytest.raises(ConfigurationError) as exc_info:
        AudioFormat(encoding="audio/pcmu", sample_rate=24000)

    assert "audio/pcm" in str(exc_info.value)


def test_a_sample_rate_other_than_24000_is_refused():
    with pytest.raises(ConfigurationError) as exc_info:
        AudioFormat(sample_rate=16000)

    assert "24000" in str(exc_info.value)


def test_the_helpers_are_frozen():
    audio_input = AudioInput()

    with pytest.raises(dataclasses.FrozenInstanceError):
        audio_input.keyterms = ["margherita"]
