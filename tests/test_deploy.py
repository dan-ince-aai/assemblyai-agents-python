"""deploy(): create once, update after, remember which agent it is."""

import pytest

from assemblyai_agents import ConfigurationError, VoiceAgent, tool
from assemblyai_agents.deploy import deploy, id_file_for, resolve_public_url

from .conftest import err


def _agent_response(agent_id: str) -> dict:
    return {
        "id": agent_id, "name": "Pizza Line", "system_prompt": "x",
        "voice": {"voice_id": "alba"}, "input": {"type": "audio"}, "output": {"type": "audio"},
        "created_at": "2026-01-01T00:00:00Z", "updated_at": "2026-01-01T00:00:00Z",
    }


@tool(timeout_seconds=8)
def lookup_order(order_id: str) -> dict:
    """Look up an order."""
    return {}


def _agent() -> VoiceAgent:
    return VoiceAgent(name="Pizza Line", voice="alba", system_prompt="x", tools=[lookup_order])


def test_first_deploy_creates_and_writes_the_id(make_client, recorder, tmp_path, monkeypatch):
    monkeypatch.delenv("AGENT_ID", raising=False)
    client = make_client([(201, _agent_response("agt_1"), None)], recorder)
    id_file = tmp_path / ".agent_id"

    agent_id = deploy(_agent(), public_url="https://agent.example.com", secret="k",
                      client=client, id_file=str(id_file), log=None)

    assert agent_id == "agt_1"
    assert recorder.requests[0].method == "POST" and recorder.requests[0].url.path == "/v1/agents"
    # Bound before it went out: the hosted tool points at the address.
    sent = recorder.requests[0].read()
    assert b"https://agent.example.com/tools/lookup_order" in sent and b"Bearer k" in sent
    assert id_file_for(str(id_file), client._config.base_url).read_text() == "agt_1"


def test_second_deploy_updates_the_stored_id(make_client, recorder, tmp_path, monkeypatch):
    monkeypatch.delenv("AGENT_ID", raising=False)
    client = make_client([(200, _agent_response("agt_1"), None)], recorder)
    id_file = tmp_path / ".agent_id"
    id_file_for(str(id_file), client._config.base_url).write_text("agt_1")

    agent_id = deploy(_agent(), public_url="https://agent.example.com", secret="k",
                      client=client, id_file=str(id_file), log=None)

    assert agent_id == "agt_1"
    assert recorder.requests[0].method == "PUT" and recorder.requests[0].url.path == "/v1/agents/agt_1"


def test_a_vanished_agent_is_recreated_and_the_id_replaced(make_client, recorder, tmp_path, monkeypatch):
    monkeypatch.delenv("AGENT_ID", raising=False)
    client = make_client(
        [(404, err("agent_not_found"), None), (201, _agent_response("agt_2"), None)], recorder
    )
    id_file = tmp_path / ".agent_id"
    path = id_file_for(str(id_file), client._config.base_url)
    path.write_text("agt_gone")

    agent_id = deploy(_agent(), public_url="https://agent.example.com", secret="k",
                      client=client, id_file=str(id_file), log=None)

    assert agent_id == "agt_2"
    assert [r.method for r in recorder.requests] == ["PUT", "POST"]
    assert path.read_text() == "agt_2"


def test_the_id_file_is_kept_per_host():
    assert id_file_for(".agent_id", "https://agents.assemblyai.com").name == ".agent_id"
    assert id_file_for(".agent_id", "https://agents.us.assemblyai.com").name == ".agent_id.agents"
    assert id_file_for(".agent_id", "https://agents.test.local").name == ".agent_id.agents"


def test_an_agent_that_hosts_nothing_needs_no_address(make_client, recorder, tmp_path, monkeypatch):
    monkeypatch.delenv("AGENT_ID", raising=False)
    monkeypatch.delenv("PUBLIC_BASE_URL", raising=False)
    client = make_client([(201, _agent_response("agt_1"), None)], recorder)
    plain = VoiceAgent(name="Pizza Line", voice="alba", system_prompt="x")

    assert deploy(plain, client=client, id_file=str(tmp_path / ".agent_id"), log=None) == "agt_1"


def test_a_hosting_agent_with_no_address_fails_before_any_request(make_client, recorder, tmp_path, monkeypatch):
    monkeypatch.delenv("PUBLIC_BASE_URL", raising=False)
    client = make_client([(201, _agent_response("agt_1"), None)], recorder)

    with pytest.raises(ConfigurationError, match="PUBLIC_BASE_URL.*Nothing in the SDK starts a tunnel"):
        deploy(_agent(), client=client, id_file=str(tmp_path / ".agent_id"), log=None)
    assert recorder.count == 0


def test_public_url_falls_back_to_the_environment(monkeypatch):
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://env.example.com/")
    assert resolve_public_url(None) == "https://env.example.com"
    assert resolve_public_url("https://arg.example.com") == "https://arg.example.com"
