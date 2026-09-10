from typing import TYPE_CHECKING, Optional

from .._pagination import AsyncPager, SyncPager
from ..models.rest import SessionListItem, SessionResponse
from ..realtime import AsyncRealtimeSession, _connect, _ws_url_from_base

if TYPE_CHECKING:
    from .._client import AsyncClient, Client


def _list_params(
    limit: Optional[int],
    cursor: Optional[str],
    status: Optional[str],
    agent_id: Optional[str],
) -> dict:
    params: dict = {}
    if limit is not None:
        params["limit"] = limit
    if cursor is not None:
        params["cursor"] = cursor
    if status is not None:
        params["status"] = status
    if agent_id is not None:
        params["agent_id"] = agent_id
    return params


class SessionsResource:
    def __init__(self, client: "Client") -> None:
        self._client = client

    def list(
        self,
        *,
        limit: Optional[int] = None,
        cursor: Optional[str] = None,
        status: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> SyncPager[SessionListItem]:
        return self._client.paginate(
            "/v1/sessions",
            item_key="sessions",
            params=_list_params(limit, cursor, status, agent_id),
            item_factory=SessionListItem.model_validate,
        )

    def get(self, session_id: str) -> SessionResponse:
        raw = self._client.request("GET", f"/v1/sessions/{session_id}")
        return SessionResponse.model_validate(raw)

    def delete(self, session_id: str) -> None:
        self._client.request_raw("DELETE", f"/v1/sessions/{session_id}")
        return None


class AsyncSessionsResource:
    def __init__(self, client: "AsyncClient") -> None:
        self._client = client

    def list(
        self,
        *,
        limit: Optional[int] = None,
        cursor: Optional[str] = None,
        status: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> AsyncPager[SessionListItem]:
        return self._client.paginate(
            "/v1/sessions",
            item_key="sessions",
            params=_list_params(limit, cursor, status, agent_id),
            item_factory=SessionListItem.model_validate,
        )

    async def get(self, session_id: str) -> SessionResponse:
        raw = await self._client.request("GET", f"/v1/sessions/{session_id}")
        return SessionResponse.model_validate(raw)

    async def delete(self, session_id: str) -> None:
        await self._client.request_raw("DELETE", f"/v1/sessions/{session_id}")
        return None

    async def connect(
        self,
        *,
        token: Optional[str] = None,
        url: Optional[str] = None,
        open_timeout: Optional[float] = 15.0,
        auto_resume: bool = False,
        max_resume_attempts: int = 5,
    ) -> AsyncRealtimeSession:
        ws_url = (
            url if url is not None else _ws_url_from_base(self._client._config.base_url)
        )
        bearer = token or self._client._config.api_key
        return await _connect(
            ws_url,
            bearer,
            open_timeout,
            auto_resume=auto_resume,
            max_resume_attempts=max_resume_attempts,
        )
