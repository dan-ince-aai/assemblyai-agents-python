from typing import TYPE_CHECKING

from ..models.rest import (
    SetToolSecretRequest,
    ToolSecretListResponse,
    ToolSecretResponse,
)

if TYPE_CHECKING:
    from .._client import AsyncClient, Client

_PATH = "/v1/tool-secrets"


def _set_body(body: SetToolSecretRequest) -> dict:
    # model_dump masks SecretStr to "**********"; the wire needs the real value.
    return {"value": body.value.get_secret_value()}


class ToolSecretsResource:
    def __init__(self, client: "Client") -> None:
        self._client = client

    def set(self, name: str, body: SetToolSecretRequest) -> ToolSecretResponse:
        raw = self._client.request("PUT", f"{_PATH}/{name}", json=_set_body(body))
        return ToolSecretResponse.model_validate(raw)

    def list(self) -> ToolSecretListResponse:
        raw = self._client.request("GET", _PATH)
        return ToolSecretListResponse.model_validate(raw)

    def delete(self, name: str) -> None:
        self._client.request("DELETE", f"{_PATH}/{name}")
        return None


class AsyncToolSecretsResource:
    def __init__(self, client: "AsyncClient") -> None:
        self._client = client

    async def set(self, name: str, body: SetToolSecretRequest) -> ToolSecretResponse:
        raw = await self._client.request("PUT", f"{_PATH}/{name}", json=_set_body(body))
        return ToolSecretResponse.model_validate(raw)

    async def list(self) -> ToolSecretListResponse:
        raw = await self._client.request("GET", _PATH)
        return ToolSecretListResponse.model_validate(raw)

    async def delete(self, name: str) -> None:
        await self._client.request("DELETE", f"{_PATH}/{name}")
        return None
