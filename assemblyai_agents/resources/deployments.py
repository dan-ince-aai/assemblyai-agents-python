from typing import TYPE_CHECKING, Optional

from .._pagination import AsyncPager, SyncPager
from ..models.rest import (
    AgentDeploymentListItem,
    AgentDeploymentResponse,
    CreateAgentDeploymentRequest,
)

if TYPE_CHECKING:
    from .._client import AsyncClient, Client

_PATH = "/v1/agent-deployments"


def _list_params(
    agent_id: Optional[str], limit: Optional[int], cursor: Optional[str]
) -> dict:
    params: dict = {}
    if agent_id is not None:
        params["agent_id"] = agent_id
    if limit is not None:
        params["limit"] = limit
    if cursor is not None:
        params["cursor"] = cursor
    return params


def _create_body(body: CreateAgentDeploymentRequest) -> dict:
    return body.model_dump(mode="json", exclude_none=True, by_alias=True)


class DeploymentsResource:
    def __init__(self, client: "Client") -> None:
        self._client = client

    def create(self, body: CreateAgentDeploymentRequest) -> AgentDeploymentResponse:
        raw = self._client.request(
            "POST", _PATH, json=_create_body(body), idempotent=True
        )
        return AgentDeploymentResponse.model_validate(raw)

    def list(
        self,
        *,
        agent_id: Optional[str] = None,
        limit: Optional[int] = None,
        cursor: Optional[str] = None,
    ) -> SyncPager[AgentDeploymentListItem]:
        return self._client.paginate(
            _PATH,
            item_key="agent_deployments",
            params=_list_params(agent_id, limit, cursor),
            item_factory=AgentDeploymentListItem.model_validate,
        )

    def get(self, deployment_id: str) -> AgentDeploymentResponse:
        raw = self._client.request("GET", f"{_PATH}/{deployment_id}")
        return AgentDeploymentResponse.model_validate(raw)

    def delete(self, deployment_id: str) -> None:
        # `request`, not `request_raw`: a refusal (still building, or still
        # serving an agent's tools) has to raise rather than read as success.
        self._client.request("DELETE", f"{_PATH}/{deployment_id}")
        return None


class AsyncDeploymentsResource:
    def __init__(self, client: "AsyncClient") -> None:
        self._client = client

    async def create(
        self, body: CreateAgentDeploymentRequest
    ) -> AgentDeploymentResponse:
        raw = await self._client.request(
            "POST", _PATH, json=_create_body(body), idempotent=True
        )
        return AgentDeploymentResponse.model_validate(raw)

    def list(
        self,
        *,
        agent_id: Optional[str] = None,
        limit: Optional[int] = None,
        cursor: Optional[str] = None,
    ) -> AsyncPager[AgentDeploymentListItem]:
        return self._client.paginate(
            _PATH,
            item_key="agent_deployments",
            params=_list_params(agent_id, limit, cursor),
            item_factory=AgentDeploymentListItem.model_validate,
        )

    async def get(self, deployment_id: str) -> AgentDeploymentResponse:
        raw = await self._client.request("GET", f"{_PATH}/{deployment_id}")
        return AgentDeploymentResponse.model_validate(raw)

    async def delete(self, deployment_id: str) -> None:
        await self._client.request("DELETE", f"{_PATH}/{deployment_id}")
        return None
