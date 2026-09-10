from typing import TYPE_CHECKING

# Module import + call-time attribute access rather than a name import, so the
# response model is resolved when it is used rather than when the package loads.
from ..models import rest

if TYPE_CHECKING:
    from .._client import AsyncClient, Client


class BuiltinToolsResource:
    def __init__(self, client: "Client") -> None:
        self._client = client

    def list(self) -> "rest.BuiltinToolListResponse":
        raw = self._client.request("GET", "/v1/builtin-tools")
        return rest.BuiltinToolListResponse.model_validate(raw)


class AsyncBuiltinToolsResource:
    def __init__(self, client: "AsyncClient") -> None:
        self._client = client

    async def list(self) -> "rest.BuiltinToolListResponse":
        raw = await self._client.request("GET", "/v1/builtin-tools")
        return rest.BuiltinToolListResponse.model_validate(raw)
