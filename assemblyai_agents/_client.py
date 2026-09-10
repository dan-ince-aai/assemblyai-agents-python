from typing import Any, Callable, Mapping, Optional

import httpx

from ._config import (
    DEFAULT_BASE_URL,
    DEFAULT_MAX_RETRIES,
    DEFAULT_TIMEOUT,
    ClientConfig,
)
from ._pagination import AsyncPager, SyncPager
from ._response import RawResponse
from ._transport import AsyncTransportCore, SyncTransportCore
from .resources.agents import AgentsResource, AsyncAgentsResource
from .resources.builtin_tools import AsyncBuiltinToolsResource, BuiltinToolsResource
from .resources.calls import AsyncCallsResource, CallsResource
from .resources.phone_numbers import AsyncPhoneNumbersResource, PhoneNumbersResource
from .resources.sessions import AsyncSessionsResource, SessionsResource
from .resources.tokens import AsyncTokensResource, TokensResource
from .resources.webhooks import AsyncWebhooksResource, WebhooksResource


def _identity(item: Any) -> Any:
    return item


class Client:
    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._config = ClientConfig.build(api_key, base_url, timeout, max_retries)
        self._http = httpx.Client(transport=transport, follow_redirects=False)
        self._core = SyncTransportCore(self._config, self._http)
        self.agents = AgentsResource(self)
        self.builtin_tools = BuiltinToolsResource(self)
        self.calls = CallsResource(self)
        self.tokens = TokensResource(self)
        self.phone_numbers = PhoneNumbersResource(self)
        self.sessions = SessionsResource(self)
        self.webhooks = WebhooksResource(self)

    def request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Mapping[str, Any]] = None,
        json: Any = None,
        headers: Optional[Mapping[str, str]] = None,
        idempotent: bool = False,
        timeout: Optional[float] = None,
    ) -> Any:
        return self._core.request(
            method,
            path,
            params=params,
            json=json,
            headers=headers,
            idempotent=idempotent,
            timeout=timeout,
        )

    def request_raw(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Mapping[str, Any]] = None,
        json: Any = None,
        headers: Optional[Mapping[str, str]] = None,
        idempotent: bool = False,
        timeout: Optional[float] = None,
    ) -> RawResponse:
        return self._core.request_raw(
            method,
            path,
            params=params,
            json=json,
            headers=headers,
            idempotent=idempotent,
            timeout=timeout,
        )

    def paginate(
        self,
        path: str,
        *,
        item_key: str,
        params: Optional[Mapping[str, Any]] = None,
        item_factory: Callable[[Any], Any] = _identity,
    ) -> SyncPager:
        return SyncPager(self.request, path, item_key, params, item_factory)

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "Client":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"Client(base_url={self._config.base_url!r})"


class AsyncClient:
    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._config = ClientConfig.build(api_key, base_url, timeout, max_retries)
        self._http = httpx.AsyncClient(transport=transport, follow_redirects=False)
        self._core = AsyncTransportCore(self._config, self._http)
        self.agents = AsyncAgentsResource(self)
        self.builtin_tools = AsyncBuiltinToolsResource(self)
        self.calls = AsyncCallsResource(self)
        self.tokens = AsyncTokensResource(self)
        self.phone_numbers = AsyncPhoneNumbersResource(self)
        self.sessions = AsyncSessionsResource(self)
        self.webhooks = AsyncWebhooksResource(self)

    async def request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Mapping[str, Any]] = None,
        json: Any = None,
        headers: Optional[Mapping[str, str]] = None,
        idempotent: bool = False,
        timeout: Optional[float] = None,
    ) -> Any:
        return await self._core.request(
            method,
            path,
            params=params,
            json=json,
            headers=headers,
            idempotent=idempotent,
            timeout=timeout,
        )

    async def request_raw(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Mapping[str, Any]] = None,
        json: Any = None,
        headers: Optional[Mapping[str, str]] = None,
        idempotent: bool = False,
        timeout: Optional[float] = None,
    ) -> RawResponse:
        return await self._core.request_raw(
            method,
            path,
            params=params,
            json=json,
            headers=headers,
            idempotent=idempotent,
            timeout=timeout,
        )

    def paginate(
        self,
        path: str,
        *,
        item_key: str,
        params: Optional[Mapping[str, Any]] = None,
        item_factory: Callable[[Any], Any] = _identity,
    ) -> AsyncPager:
        return AsyncPager(self.request, path, item_key, params, item_factory)

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> "AsyncClient":
        return self

    async def __aexit__(self, *exc_info) -> None:
        await self.aclose()

    def __repr__(self) -> str:
        return f"AsyncClient(base_url={self._config.base_url!r})"
