from typing import TYPE_CHECKING, Optional, Union

from .._agent import VoiceAgent
from .._pagination import AsyncPager, SyncPager
from ..models.rest import (
    AgentCreateRequest,
    AgentListItem,
    AgentResponse,
    AgentUpdateRequest,
)

if TYPE_CHECKING:
    from .._client import AsyncClient, Client

CreateBody = Union[VoiceAgent, AgentCreateRequest]
UpdateBody = Union[VoiceAgent, AgentUpdateRequest]


def _create_payload(body: CreateBody) -> dict:
    request = body.to_request() if isinstance(body, VoiceAgent) else body
    return request.model_dump(mode="json", exclude_none=True, by_alias=True)


def _update_payload(body: UpdateBody) -> dict:
    # A `VoiceAgent` goes out whole. `PUT /v1/agents/{id}` replaces the stored
    # agent rather than merging into it, so a partial body drops whatever it
    # omits; the raw `AgentUpdateRequest` path stays open for callers who mean
    # exactly that.
    request = body.to_update_request() if isinstance(body, VoiceAgent) else body
    return request.model_dump(mode="json", exclude_none=True, by_alias=True)


class AgentsResource:
    def __init__(self, client: "Client") -> None:
        self._client = client

    def create(self, body: CreateBody) -> AgentResponse:
        raw = self._client.request("POST", "/v1/agents", json=_create_payload(body))
        return AgentResponse.model_validate(raw)

    def list(
        self, *, limit: Optional[int] = None, cursor: Optional[str] = None
    ) -> SyncPager[AgentListItem]:
        params = {}
        if limit is not None:
            params["limit"] = limit
        if cursor is not None:
            params["cursor"] = cursor
        return self._client.paginate(
            "/v1/agents",
            item_key="agents",
            params=params,
            item_factory=AgentListItem.model_validate,
        )

    def get(self, agent_id: str) -> AgentResponse:
        raw = self._client.request("GET", f"/v1/agents/{agent_id}")
        return AgentResponse.model_validate(raw)

    def update(self, agent_id: str, body: UpdateBody) -> AgentResponse:
        raw = self._client.request(
            "PUT", f"/v1/agents/{agent_id}", json=_update_payload(body)
        )
        return AgentResponse.model_validate(raw)

    def delete(self, agent_id: str) -> None:
        self._client.request_raw("DELETE", f"/v1/agents/{agent_id}")
        return None


class AsyncAgentsResource:
    def __init__(self, client: "AsyncClient") -> None:
        self._client = client

    async def create(self, body: CreateBody) -> AgentResponse:
        raw = await self._client.request(
            "POST", "/v1/agents", json=_create_payload(body)
        )
        return AgentResponse.model_validate(raw)

    def list(
        self, *, limit: Optional[int] = None, cursor: Optional[str] = None
    ) -> AsyncPager[AgentListItem]:
        params = {}
        if limit is not None:
            params["limit"] = limit
        if cursor is not None:
            params["cursor"] = cursor
        return self._client.paginate(
            "/v1/agents",
            item_key="agents",
            params=params,
            item_factory=AgentListItem.model_validate,
        )

    async def get(self, agent_id: str) -> AgentResponse:
        raw = await self._client.request("GET", f"/v1/agents/{agent_id}")
        return AgentResponse.model_validate(raw)

    async def update(self, agent_id: str, body: UpdateBody) -> AgentResponse:
        raw = await self._client.request(
            "PUT", f"/v1/agents/{agent_id}", json=_update_payload(body)
        )
        return AgentResponse.model_validate(raw)

    async def delete(self, agent_id: str) -> None:
        await self._client.request_raw("DELETE", f"/v1/agents/{agent_id}")
        return None
