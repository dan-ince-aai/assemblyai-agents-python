import json

import pytest
from assemblyai_agents import ConflictError, NotFoundError
from assemblyai_agents.models.rest import (
    AgentDeploymentListItem,
    AgentDeploymentResponse,
    CreateAgentDeploymentRequest,
    DeploymentStatus,
    DeploymentType,
    SetToolSecretRequest,
    ToolSecretListResponse,
    ToolSecretResponse,
)

from .conftest import err

TS = "2026-09-18T10:00:00Z"
AGENT_ID = "agent_1"


def _deployment(deployment_id: str = "agentdep_1", status: str = "pending") -> dict:
    return {
        "id": deployment_id,
        "agent_id": AGENT_ID,
        "deployment_type": "tools",
        "status": status,
        "created_at": TS,
        "updated_at": TS,
    }


def _page(ids: list, cursor: str = "") -> dict:
    return {
        "agent_deployments": [_deployment(i, "ready") for i in ids],
        "has_more": bool(cursor),
        "response_metadata": {"next_cursor": cursor},
    }


def test_create_posts_one_code_field_with_an_idempotency_key(make_client, recorder):
    client = make_client([(201, _deployment(), None)], recorder)

    result = client.deployments.create(
        CreateAgentDeploymentRequest(
            agent_id=AGENT_ID,
            deployment_type=DeploymentType.service,
            archive="YQ==",
        )
    )

    request = recorder.requests[0]
    assert request.method == "POST"
    assert request.url.path == "/v1/agent-deployments"
    assert json.loads(request.content) == {
        "agent_id": AGENT_ID,
        "deployment_type": "service",
        "archive": "YQ==",
    }
    assert request.headers.get("Idempotency-Key")
    assert isinstance(result, AgentDeploymentResponse)
    assert result.status is DeploymentStatus.pending


def test_list_filters_by_agent_and_follows_the_cursor(make_client, recorder):
    client = make_client(
        [(200, _page(["d1"], cursor="c1"), None), (200, _page(["d2"]), None)],
        recorder,
    )

    items = list(client.deployments.list(agent_id=AGENT_ID, limit=1))

    assert [i.id for i in items] == ["d1", "d2"]
    assert all(isinstance(i, AgentDeploymentListItem) for i in items)
    first, second = recorder.requests
    assert first.url.params.get("agent_id") == AGENT_ID
    assert first.url.params.get("limit") == "1"
    assert second.url.params.get("cursor") == "c1"
    assert second.url.params.get("agent_id") == AGENT_ID


def test_get_reads_the_single_deployment_route(make_client, recorder):
    client = make_client([(200, _deployment("agentdep_9", "ready"), None)], recorder)

    result = client.deployments.get("agentdep_9")

    assert recorder.requests[0].url.path == "/v1/agent-deployments/agentdep_9"
    assert result.status is DeploymentStatus.ready


def test_delete_raises_when_the_server_refuses(make_client, recorder):
    client = make_client([(409, err("deployment_in_use"), None)], recorder)

    with pytest.raises(ConflictError):
        client.deployments.delete("agentdep_1")

    assert recorder.requests[0].method == "DELETE"


def test_delete_returns_none_on_204(make_client, recorder):
    client = make_client([(204, None, None)], recorder)

    assert client.deployments.delete("agentdep_1") is None


def test_setting_a_secret_sends_the_real_value_not_the_mask(make_client, recorder):
    client = make_client(
        [(200, {"name": "orders_key", "created_at": TS, "updated_at": TS}, None)],
        recorder,
    )

    result = client.tool_secrets.set("orders_key", SetToolSecretRequest(value="s3cr3t"))

    request = recorder.requests[0]
    assert request.method == "PUT"
    assert request.url.path == "/v1/tool-secrets/orders_key"
    assert json.loads(request.content) == {"value": "s3cr3t"}
    assert isinstance(result, ToolSecretResponse)


def test_listing_secrets_is_one_unpaginated_read(make_client, recorder):
    body = {"secrets": [{"name": "a", "created_at": TS, "updated_at": TS}]}
    client = make_client([(200, body, None)], recorder)

    result = client.tool_secrets.list()

    assert recorder.count == 1
    assert recorder.requests[0].url.path == "/v1/tool-secrets"
    assert isinstance(result, ToolSecretListResponse)
    assert [s.name for s in result.secrets] == ["a"]


def test_deleting_a_missing_secret_raises(make_client, recorder):
    client = make_client([(404, err("tool_secret_not_found"), None)], recorder)

    with pytest.raises(NotFoundError):
        client.tool_secrets.delete("gone")
