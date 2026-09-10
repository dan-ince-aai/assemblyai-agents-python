import dataclasses

import pytest
from assemblyai_agents import (
    AudioFormat,
    AudioInput,
    AudioOutput,
    ConfigurationError,
    VoiceAgent,
    tool,
)
from assemblyai_agents.models.rest import (
    AgentCreateRequest,
    AgentUpdateRequest,
    LlmConfigRequest,
    VoiceConfig,
)

PROMPT = "You take pizza orders."


def declare_lookup_order():
    @tool
    async def lookup_order(order_id: str) -> dict:
        """Look up one of the caller's orders by its ID."""
        return {"order_id": order_id}

    return lookup_order


def test_a_minimal_declaration_builds_the_wire_model_exactly():
    agent = VoiceAgent(name="Pizza Line", voice="ivy", system_prompt=PROMPT)

    assert agent.to_request() == AgentCreateRequest(
        name="Pizza Line",
        system_prompt=PROMPT,
        voice=VoiceConfig(voice_id="ivy"),
    )


def test_a_voice_string_becomes_the_nested_voice_config():
    request = VoiceAgent(
        name="Pizza Line", voice="james", system_prompt=PROMPT
    ).to_request()

    assert request.voice == VoiceConfig(voice_id="james")


def test_a_single_llm_config_becomes_the_one_element_list_the_wire_takes():
    llm = LlmConfigRequest(
        base_url="https://api.openai.com/v1", model="gpt-4o-mini", api_key="k"
    )

    request = VoiceAgent(
        name="Pizza Line", voice="ivy", system_prompt=PROMPT, llm=llm
    ).to_request()

    assert request.llm == [llm]


def test_a_fully_populated_declaration_builds_the_wire_model_exactly():
    llm = LlmConfigRequest(
        base_url="https://api.openai.com/v1", model="gpt-4o-mini", api_key="k"
    )
    lookup_order = declare_lookup_order()

    agent = VoiceAgent(
        name="Pizza Line",
        voice="ivy",
        system_prompt=PROMPT,
        greeting="Pizza Palace — what can I get you?",
        llm=llm,
        input=AudioInput(
            format=AudioFormat(encoding="audio/pcm", sample_rate=24000),
            keyterms=["margherita", "calzone"],
        ),
        output=AudioOutput(volume=90.0),
        tools=[lookup_order],
    )

    assert agent.to_request() == AgentCreateRequest(
        name="Pizza Line",
        system_prompt=PROMPT,
        greeting="Pizza Palace — what can I get you?",
        voice=VoiceConfig(voice_id="ivy"),
        input={
            "type": "audio",
            "format": {"encoding": "audio/pcm", "sample_rate": 24000},
            "keyterms": ["margherita", "calzone"],
        },
        output={"type": "audio", "volume": 90.0},
        tools=[lookup_order.definition()],
        llm=[llm],
    )


def test_the_declaration_carries_the_tools_it_was_given():
    lookup_order = declare_lookup_order()

    agent = VoiceAgent(
        name="Pizza Line", voice="ivy", system_prompt=PROMPT, tools=[lookup_order]
    )

    assert agent.tools == [lookup_order]
    assert agent.to_request().tools[0].name == "lookup_order"


def test_two_tools_under_one_name_are_refused():
    # The server refuses this too; refusing it here turns a round trip into an
    # immediate error.
    first = declare_lookup_order()
    second = declare_lookup_order()

    with pytest.raises(ConfigurationError) as exc_info:
        VoiceAgent(
            name="Pizza Line",
            voice="ivy",
            system_prompt=PROMPT,
            tools=[first, second],
        )

    assert "lookup_order" in str(exc_info.value)
    assert "twice" in str(exc_info.value)


def test_keyterms_is_not_a_top_level_shortcut():
    # It lives inside the input config on the wire, and a shortcut that
    # relocates a field is the parallel vocabulary this builder removes.
    with pytest.raises(TypeError):
        VoiceAgent(
            name="Pizza Line",
            voice="ivy",
            system_prompt=PROMPT,
            keyterms=["margherita"],
        )


def test_the_update_request_sends_the_whole_declaration():
    # `PUT /v1/agents/{id}` replaces rather than merges, so a partial update
    # would silently drop everything it left out.
    agent = VoiceAgent(
        name="Pizza Line",
        voice="ivy",
        system_prompt=PROMPT,
        greeting="Pizza Palace — what can I get you?",
    )

    assert agent.to_update_request() == AgentUpdateRequest(
        name="Pizza Line",
        system_prompt=PROMPT,
        greeting="Pizza Palace — what can I get you?",
        voice=VoiceConfig(voice_id="ivy"),
    )


def test_the_system_prompt_is_dedented_and_stripped():
    agent = VoiceAgent(
        name="Pizza Line",
        voice="ivy",
        system_prompt="""
        You take pizza orders. Answer in one or two sentences.
            Confirm the address before you finish.
        """,
    )

    # The block's own indent goes; the relative indent inside it is content.
    assert agent.system_prompt == (
        "You take pizza orders. Answer in one or two sentences.\n"
        "    Confirm the address before you finish."
    )
    assert agent.to_request().system_prompt == agent.system_prompt


def test_voice_is_required():
    with pytest.raises(TypeError) as exc_info:
        VoiceAgent(name="Pizza Line", system_prompt=PROMPT)

    assert "voice" in str(exc_info.value)


def test_subclassing_raises_at_class_creation():
    with pytest.raises(ConfigurationError) as exc_info:

        class MyAgent(VoiceAgent):
            pass

    message = str(exc_info.value)
    assert "subclasses VoiceAgent" in message
    assert "VoiceAgent(...)" in message


def test_every_field_is_keyword_only():
    with pytest.raises(TypeError):
        VoiceAgent("Pizza Line", PROMPT, "ivy")


def test_a_declaration_is_frozen():
    agent = VoiceAgent(name="Pizza Line", voice="ivy", system_prompt=PROMPT)

    with pytest.raises(dataclasses.FrozenInstanceError):
        agent.voice = "james"


def test_the_optional_fields_default_to_unset():
    agent = VoiceAgent(name="Pizza Line", voice="ivy", system_prompt=PROMPT)

    assert agent.greeting is None
    assert agent.llm is None
    assert agent.input is None
    assert agent.output is None
    assert agent.tools is None


def test_the_same_name_may_be_declared_twice_in_one_process():
    # Declaring is no longer registration, so two agents built in one process
    # (a test parametrised over voices, say) no longer collide.
    VoiceAgent(name="Pizza Line", voice="ivy", system_prompt=PROMPT)
    VoiceAgent(name="Pizza Line", voice="james", system_prompt=PROMPT)
