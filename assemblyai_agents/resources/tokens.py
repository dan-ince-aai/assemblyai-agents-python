from typing import TYPE_CHECKING, Optional

from ..models.rest import TokenCreateRequest, TokenResponse

if TYPE_CHECKING:
    from .._client import AsyncClient, Client


class TokensResource:
    def __init__(self, client: "Client") -> None:
        self._client = client

    def create(self, body: Optional[TokenCreateRequest] = None) -> TokenResponse:
        json = (
            None
            if body is None
            else body.model_dump(mode="json", exclude_none=True, by_alias=True)
        )
        raw = self._client.request("POST", "/v1/tokens", json=json)
        return TokenResponse.model_validate(raw)


class AsyncTokensResource:
    def __init__(self, client: "AsyncClient") -> None:
        self._client = client

    async def create(self, body: Optional[TokenCreateRequest] = None) -> TokenResponse:
        json = (
            None
            if body is None
            else body.model_dump(mode="json", exclude_none=True, by_alias=True)
        )
        raw = await self._client.request("POST", "/v1/tokens", json=json)
        return TokenResponse.model_validate(raw)
