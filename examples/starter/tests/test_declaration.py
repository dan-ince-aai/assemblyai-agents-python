"""What the deployed agent looks like on the wire."""

import agent as agent_module
from assemblyai_agents.models.rest import HttpMethod


def test_every_tool_is_hosted_so_a_phone_call_can_be_answered():
    # A client-resident tool has nothing to answer it on a phone call, and the
    # SDK refuses to attach a number to an agent that still has one.
    assert agent_module.agent.client_resident_tool_names() == ()
    for tool in agent_module.agent.to_request().tools:
        assert tool.http is not None
        assert tool.http.url.startswith("https://")
        assert tool.http.http_method == HttpMethod.POST


def test_the_tools_take_what_the_caller_said_rather_than_a_derived_value():
    # The platform refuses an argument whose value the conversation never
    # established, so anything the caller speaks is passed through verbatim and
    # read at the backend.
    tools = {tool.name: tool for tool in agent_module.agent.to_request().tools}

    assert "caller_said" in tools["verify_caller"].parameters["properties"]
    assert "reference_said" in tools["find_patient"].parameters["properties"]


def test_the_lookup_hands_back_nothing_identifying():
    # The caller states their own name; the tool checks it. A lookup that
    # returned the name would let the agent greet an unverified caller by it.
    import asyncio

    result = asyncio.run(agent_module.find_patient.invoke(reference_said="four four seven one"))

    assert result["found"] is True
    joined = str(result).lower()
    assert "delgado" not in joined and "maria" not in joined


def test_the_pre_connect_lookup_may_rewrite_the_greeting():
    entry = agent_module.agent.to_request().pre_connect_requests[0]

    assert entry.allow_overrides == ["greeting"]
    assert entry.timeout_ms <= 800
    assert {captured.name for captured in entry.returns} == {"reference", "first_name"}


def test_replies_come_from_our_own_endpoint():
    llm = agent_module.agent.to_request().llm[0]
    assert llm.base_url.endswith("/v1")


def test_the_greeting_names_the_practice_and_the_recording():
    greeting = agent_module.agent.greeting
    assert "Fairview Dental" in greeting
    assert "recorded line" in greeting


def test_expose_is_the_same_file_as_the_examples_copy():
    """The starter ships its own `expose.py` so a copied project runs.

    Two copies drift, so this pins them together. If the examples copy changes,
    copy it over; when agent code can be deployed directly, both are deleted.
    """
    from pathlib import Path

    here = Path(__file__).resolve().parent.parent
    upstream = here.parent / "expose.py"
    if not upstream.exists():
        return  # this project has been copied out of the SDK repo; nothing to pin
    assert (here / "expose.py").read_text() == upstream.read_text()
