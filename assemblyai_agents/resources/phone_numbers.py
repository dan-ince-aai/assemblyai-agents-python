import warnings
from typing import TYPE_CHECKING, Optional
from urllib.parse import quote

from .._agent import VoiceAgent
from .._pagination import AsyncPager, SyncPager
from ..models.rest import (
    ImportPhoneNumberRequest,
    PhoneNumberAssignAgentRequest,
    PhoneNumberResponse,
    PurchaseAvailablePhoneNumberRequest,
    PurchasePhoneNumberRequest,
)

if TYPE_CHECKING:
    from .._client import AsyncClient, Client


def _warn_agent_kwarg(agent: Optional[VoiceAgent]) -> None:
    if agent is None:
        return
    warnings.warn(
        "assign_agent(agent=...) no longer does anything: every tool on a declaration "
        "is served over HTTPS, so there is nothing to refuse. Drop the argument.",
        DeprecationWarning,
        stacklevel=3,
    )


def _list_params(limit: Optional[int], cursor: Optional[str]) -> dict:
    params: dict = {}
    if limit is not None:
        params["limit"] = limit
    if cursor is not None:
        params["cursor"] = cursor
    return params


class PhoneNumbersResource:
    def __init__(self, client: "Client") -> None:
        self._client = client

    def list(
        self, *, limit: Optional[int] = None, cursor: Optional[str] = None
    ) -> SyncPager[PhoneNumberResponse]:
        return self._client.paginate(
            "/v1/phone-numbers",
            item_key="phone_numbers",
            params=_list_params(limit, cursor),
            item_factory=PhoneNumberResponse.model_validate,
        )

    def purchase_available(
        self, body: PurchaseAvailablePhoneNumberRequest
    ) -> PhoneNumberResponse:
        raw = self._client.request(
            "POST",
            "/v1/phone-numbers",
            json=body.model_dump(mode="json", exclude_none=True, by_alias=True),
            idempotent=True,
        )
        return PhoneNumberResponse.model_validate(raw)

    def purchase(self, body: PurchasePhoneNumberRequest) -> None:
        self._client.request_raw(
            "POST",
            "/v1/phone-numbers/purchase",
            json=body.model_dump(mode="json", exclude_none=True, by_alias=True),
            idempotent=True,
        )
        return None

    def import_(self, body: ImportPhoneNumberRequest) -> None:
        self._client.request_raw(
            "POST",
            "/v1/phone-numbers/import",
            json=body.model_dump(mode="json", exclude_none=True, by_alias=True),
            idempotent=True,
        )
        return None

    def get(self, number: str) -> PhoneNumberResponse:
        raw = self._client.request("GET", f"/v1/phone-numbers/{quote(number, safe='')}")
        return PhoneNumberResponse.model_validate(raw)

    def deregister(self, number: str) -> None:
        self._client.request_raw(
            "DELETE", f"/v1/phone-numbers/{quote(number, safe='')}"
        )
        return None

    def assign_agent(
        self,
        number: str,
        body: PhoneNumberAssignAgentRequest,
        *,
        agent: Optional[VoiceAgent] = None,
    ) -> None:
        _warn_agent_kwarg(agent)
        self._client.request_raw(
            "PUT",
            f"/v1/phone-numbers/{quote(number, safe='')}/agent",
            json=body.model_dump(mode="json", exclude_none=True, by_alias=True),
        )
        return None

    def unassign_agent(self, number: str) -> None:
        self._client.request_raw(
            "DELETE", f"/v1/phone-numbers/{quote(number, safe='')}/agent"
        )
        return None


class AsyncPhoneNumbersResource:
    def __init__(self, client: "AsyncClient") -> None:
        self._client = client

    def list(
        self, *, limit: Optional[int] = None, cursor: Optional[str] = None
    ) -> AsyncPager[PhoneNumberResponse]:
        return self._client.paginate(
            "/v1/phone-numbers",
            item_key="phone_numbers",
            params=_list_params(limit, cursor),
            item_factory=PhoneNumberResponse.model_validate,
        )

    async def purchase_available(
        self, body: PurchaseAvailablePhoneNumberRequest
    ) -> PhoneNumberResponse:
        raw = await self._client.request(
            "POST",
            "/v1/phone-numbers",
            json=body.model_dump(mode="json", exclude_none=True, by_alias=True),
            idempotent=True,
        )
        return PhoneNumberResponse.model_validate(raw)

    async def purchase(self, body: PurchasePhoneNumberRequest) -> None:
        await self._client.request_raw(
            "POST",
            "/v1/phone-numbers/purchase",
            json=body.model_dump(mode="json", exclude_none=True, by_alias=True),
            idempotent=True,
        )
        return None

    async def import_(self, body: ImportPhoneNumberRequest) -> None:
        await self._client.request_raw(
            "POST",
            "/v1/phone-numbers/import",
            json=body.model_dump(mode="json", exclude_none=True, by_alias=True),
            idempotent=True,
        )
        return None

    async def get(self, number: str) -> PhoneNumberResponse:
        raw = await self._client.request(
            "GET", f"/v1/phone-numbers/{quote(number, safe='')}"
        )
        return PhoneNumberResponse.model_validate(raw)

    async def deregister(self, number: str) -> None:
        await self._client.request_raw(
            "DELETE", f"/v1/phone-numbers/{quote(number, safe='')}"
        )
        return None

    async def assign_agent(
        self,
        number: str,
        body: PhoneNumberAssignAgentRequest,
        *,
        agent: Optional[VoiceAgent] = None,
    ) -> None:
        _warn_agent_kwarg(agent)
        await self._client.request_raw(
            "PUT",
            f"/v1/phone-numbers/{quote(number, safe='')}/agent",
            json=body.model_dump(mode="json", exclude_none=True, by_alias=True),
        )
        return None

    async def unassign_agent(self, number: str) -> None:
        await self._client.request_raw(
            "DELETE", f"/v1/phone-numbers/{quote(number, safe='')}/agent"
        )
        return None
