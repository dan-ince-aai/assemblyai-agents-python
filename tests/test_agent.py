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
    HttpToolHeaderInput,
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

    with pytest.warns(DeprecationWarning):
        request = VoiceAgent(
            name="Pizza Line", voice="alba", system_prompt=PROMPT, llm=llm
        ).to_request()

    assert request.llm == [llm]


def test_a_fully_populated_declaration_builds_the_wire_model_exactly():
    lookup_order = declare_lookup_order()

    def decide(turn):
        return None

    agent = VoiceAgent(
        name="Pizza Line",
        voice="alba",
        system_prompt=PROMPT,
        greeting="Pizza Palace — what can I get you?",
        reply=decide,
        input=AudioInput(
            format=AudioFormat(encoding="audio/pcm", sample_rate=24000),
            keyterms=["margherita", "calzone"],
        ),
        output=AudioOutput(volume=90.0),
        tools=[lookup_order],
        public_url="https://agent.example.com",
        secret="k",
    )

    auth = [HttpToolHeaderInput(name="Authorization", value="Bearer k")]
    assert agent.to_request() == AgentCreateRequest(
        name="Pizza Line",
        system_prompt=PROMPT,
        greeting="Pizza Palace — what can I get you?",
        voice=VoiceConfig(voice_id="alba"),
        input={
            "type": "audio",
            "format": {"encoding": "audio/pcm", "sample_rate": 24000},
            "keyterms": ["margherita", "calzone"],
        },
        output={"type": "audio", "volume": 90.0},
        # The bare tool is hosted here, so it goes out pointed at this process.
        tools=[
            lookup_order.hosted_at(
                "https://agent.example.com/tools/lookup_order", headers=auth
            ).definition()
        ],
        # `reply=` is the one-element llm list pointing back at this process.
        llm=[LlmConfigRequest(base_url="https://agent.example.com/v1", model="pizza-line", api_key="k")],
    )


def test_the_declaration_carries_the_tools_it_was_given():
    lookup_order = declare_lookup_order()

    agent = VoiceAgent(
        name="Pizza Line", voice="alba", system_prompt=PROMPT, tools=[lookup_order]
    )

    assert agent.tools == [lookup_order]
    assert agent.hosted_tool_names() == ("lookup_order",)
    bound = agent.hosted_at("https://agent.example.com")
    assert bound.to_request().tools[0].name == "lookup_order"
    assert bound.to_request().tools[0].http.url == "https://agent.example.com/tools/lookup_order"


def test_a_hosted_tool_needs_an_address_before_it_can_go_on_the_wire():
    # A bare tool is served by this process, and the platform has to be told
    # where that is. Refusing here beats a 422 after the round trip.
    agent = VoiceAgent(
        name="Pizza Line", voice="alba", system_prompt=PROMPT, tools=[declare_lookup_order()]
    )
    assert agent.needs_address is True
    with pytest.raises(ConfigurationError, match="hosted tools \\(`lookup_order`\\).*PUBLIC_BASE_URL"):
        agent.to_request()


def test_an_external_tool_needs_no_address():
    @tool(url="https://api.example.com/weather")
    def weather(city: str) -> dict:
        """Weather for a city."""
        return {}

    agent = VoiceAgent(name="Pizza Line", voice="alba", system_prompt=PROMPT, tools=[weather])
    assert agent.needs_address is False
    assert agent.to_request().tools[0].http.url == "https://api.example.com/weather"


def test_reply_becomes_the_llm_list_pointing_at_this_process():
    agent = VoiceAgent(
        name="Pizza Line", voice="alba", system_prompt=PROMPT, reply=lambda turn: None,
        public_url="https://agent.example.com/", secret="k",
    )
    assert agent.hosts_replies is True
    assert agent.public_url == "https://agent.example.com"          # trailing slash dropped
    assert agent.to_request().llm == [
        LlmConfigRequest(base_url="https://agent.example.com/v1", model="pizza-line", api_key="k")
    ]


def test_no_reply_means_the_platforms_model_talks():
    agent = VoiceAgent(name="Pizza Line", voice="alba", system_prompt=PROMPT)
    assert agent.hosts_replies is False
    assert agent.needs_address is False
    assert agent.to_request().llm is None


def test_reply_and_llm_together_are_refused():
    with pytest.raises(ConfigurationError, match="two modes"):
        VoiceAgent(
            name="Pizza Line", voice="alba", system_prompt=PROMPT, reply=lambda turn: None,
            llm=LlmConfigRequest(base_url="https://api.openai.com/v1", model="m", api_key="k"),
        )


def test_llm_is_deprecated_but_still_goes_on_the_wire():
    llm = LlmConfigRequest(base_url="https://api.openai.com/v1", model="m", api_key="k")
    with pytest.warns(DeprecationWarning, match="llm=.*deprecated"):
        agent = VoiceAgent(name="Pizza Line", voice="alba", system_prompt=PROMPT, llm=llm)
    assert agent.to_request().llm == [llm]


def test_a_reply_that_is_not_callable_is_refused():
    with pytest.raises(ConfigurationError, match="not callable"):
        VoiceAgent(name="Pizza Line", voice="alba", system_prompt=PROMPT, reply="decide")


def test_a_public_url_that_is_not_https_is_refused():
    with pytest.raises(ConfigurationError, match="not https"):
        VoiceAgent(name="Pizza Line", voice="alba", system_prompt=PROMPT, public_url="http://x")


def test_the_secret_never_appears_in_the_repr():
    agent = VoiceAgent(
        name="Pizza Line", voice="alba", system_prompt=PROMPT, public_url="https://x.example.com", secret="s3cret"
    )
    assert "s3cret" not in repr(agent)


def test_hosted_at_returns_a_bound_copy_and_leaves_the_original_alone():
    agent = VoiceAgent(name="Pizza Line", voice="alba", system_prompt=PROMPT, tools=[declare_lookup_order()])
    bound = agent.hosted_at("https://agent.example.com", secret="k")
    assert bound.public_url == "https://agent.example.com" and bound.secret == "k"
    assert agent.public_url is None and agent.secret is None
    assert bound.tools == agent.tools                      # the declaration itself is shared


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
