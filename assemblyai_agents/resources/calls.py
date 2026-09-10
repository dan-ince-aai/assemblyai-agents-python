from typing import TYPE_CHECKING, Optional

from .._pagination import AsyncPager, SyncPager
from ..models.rest import (
    CallDirection,
    CallGetResponse,
    CallListItem,
    CallResponse,
    CallStatus,
    CreateCallRequest,
)

if TYPE_CHECKING:
    from .._client import AsyncClient, Client


def _list_params(
    limit: Optional[int],
    cursor: Optional[str],
    status: Optional[CallStatus],
    direction: Optional[CallDirection],
) -> dict:
    params: dict = {}
    if limit is not None:
        params["limit"] = limit
    if cursor is not None:
        params["cursor"] = cursor
    if status is not None:
        params["status"] = status.value
    if direction is not None:
        params["direction"] = direction.value
    return params


class CallsResource:
    def __init__(self, client: "Client") -> None:
        self._client = client

    def list(
        self,
        *,
        limit: Optional[int] = None,
        cursor: Optional[str] = None,
        status: Optional[CallStatus] = None,
        direction: Optional[CallDirection] = None,
    ) -> SyncPager[CallListItem]:
        return self._client.paginate(
            "/v1/calls",
            item_key="calls",
            params=_list_params(limit, cursor, status, direction),
            item_factory=CallListItem.model_validate,
        )

    def create(self, body: CreateCallRequest) -> CallResponse:
        raw = self._client.request(
            "POST",
            "/v1/calls",
            json=body.model_dump(mode="json", exclude_none=True, by_alias=True),
            idempotent=True,
        )
        return CallResponse.model_validate(raw)

    def get(self, call_id: str) -> CallGetResponse:
        raw = self._client.request("GET", f"/v1/calls/{call_id}")
        return CallGetResponse.model_validate(raw)

    def delete(self, call_id: str) -> None:
        self._client.request_raw("DELETE", f"/v1/calls/{call_id}")
        return None


class AsyncCallsResource:
    def __init__(self, client: "AsyncClient") -> None:
        self._client = client

    def list(
        self,
        *,
        limit: Optional[int] = None,
        cursor: Optional[str] = None,
        status: Optional[CallStatus] = None,
        direction: Optional[CallDirection] = None,
    ) -> AsyncPager[CallListItem]:
        return self._client.paginate(
            "/v1/calls",
            item_key="calls",
            params=_list_params(limit, cursor, status, direction),
            item_factory=CallListItem.model_validate,
        )

    async def create(self, body: CreateCallRequest) -> CallResponse:
        raw = await self._client.request(
            "POST",
            "/v1/calls",
            json=body.model_dump(mode="json", exclude_none=True, by_alias=True),
            idempotent=True,
        )
        return CallResponse.model_validate(raw)

    async def get(self, call_id: str) -> CallGetResponse:
        raw = await self._client.request("GET", f"/v1/calls/{call_id}")
        return CallGetResponse.model_validate(raw)

    async def delete(self, call_id: str) -> None:
        await self._client.request_raw("DELETE", f"/v1/calls/{call_id}")
        return None
