from typing import TYPE_CHECKING, Optional

from .._pagination import AsyncPager, SyncPager
from ..models.rest import (
    CreateWebhookSubscriptionRequest,
    UpdateWebhookSubscriptionRequest,
    WebhookDeliveryListResponse,
    WebhookDeliveryResponse,
    WebhookSubscriptionResponse,
)

if TYPE_CHECKING:
    from .._client import AsyncClient, Client


def _subscription_list_params(
    limit: Optional[int], cursor: Optional[str], include_disabled: Optional[bool]
) -> dict:
    params: dict = {}
    if limit is not None:
        params["limit"] = limit
    if cursor is not None:
        params["cursor"] = cursor
    if include_disabled is not None:
        params["include_disabled"] = include_disabled
    return params


def _delivery_list_params(limit: Optional[int], cursor: Optional[str]) -> dict:
    params: dict = {}
    if limit is not None:
        params["limit"] = limit
    if cursor is not None:
        params["cursor"] = cursor
    return params


def _create_body(body: CreateWebhookSubscriptionRequest) -> dict:
    dumped = body.model_dump(mode="json", exclude_none=True, by_alias=True)
    # model_dump masks SecretStr to "**********"; the wire needs the real secret.
    dumped["secret"] = body.secret.get_secret_value()
    return dumped


def _update_body(body: UpdateWebhookSubscriptionRequest) -> dict:
    dumped = body.model_dump(mode="json", exclude_none=True, by_alias=True)
    if body.secret is not None:
        dumped["secret"] = body.secret.get_secret_value()
    return dumped


class WebhooksResource:
    def __init__(self, client: "Client") -> None:
        self._client = client

    def create(
        self, body: CreateWebhookSubscriptionRequest
    ) -> WebhookSubscriptionResponse:
        raw = self._client.request(
            "POST", "/v1/webhook-subscriptions", json=_create_body(body)
        )
        return WebhookSubscriptionResponse.model_validate(raw)

    def list(
        self,
        *,
        limit: Optional[int] = None,
        cursor: Optional[str] = None,
        include_disabled: Optional[bool] = None,
    ) -> SyncPager[WebhookSubscriptionResponse]:
        return self._client.paginate(
            "/v1/webhook-subscriptions",
            item_key="subscriptions",
            params=_subscription_list_params(limit, cursor, include_disabled),
            item_factory=WebhookSubscriptionResponse.model_validate,
        )

    def get(self, subscription_id: str) -> WebhookSubscriptionResponse:
        raw = self._client.request(
            "GET", f"/v1/webhook-subscriptions/{subscription_id}"
        )
        return WebhookSubscriptionResponse.model_validate(raw)

    def update(
        self, subscription_id: str, body: UpdateWebhookSubscriptionRequest
    ) -> WebhookSubscriptionResponse:
        raw = self._client.request(
            "PATCH",
            f"/v1/webhook-subscriptions/{subscription_id}",
            json=_update_body(body),
        )
        return WebhookSubscriptionResponse.model_validate(raw)

    def delete(self, subscription_id: str) -> None:
        self._client.request_raw(
            "DELETE", f"/v1/webhook-subscriptions/{subscription_id}"
        )
        return None

    def list_deliveries(
        self,
        session_id: str,
        *,
        limit: Optional[int] = None,
        cursor: Optional[str] = None,
    ) -> SyncPager[WebhookDeliveryResponse]:
        return self._client.paginate(
            f"/v1/sessions/{session_id}/webhook-deliveries",
            item_key="deliveries",
            params=_delivery_list_params(limit, cursor),
            item_factory=WebhookDeliveryResponse.model_validate,
        )

    def list_latest_deliveries(self, session_id: str) -> WebhookDeliveryListResponse:
        """The most recent delivery attempt per subscription; nothing is re-sent."""
        raw = self._client.request(
            "GET", f"/v1/sessions/{session_id}/webhook-deliveries/latest"
        )
        return WebhookDeliveryListResponse.model_validate(raw)


class AsyncWebhooksResource:
    def __init__(self, client: "AsyncClient") -> None:
        self._client = client

    async def create(
        self, body: CreateWebhookSubscriptionRequest
    ) -> WebhookSubscriptionResponse:
        raw = await self._client.request(
            "POST", "/v1/webhook-subscriptions", json=_create_body(body)
        )
        return WebhookSubscriptionResponse.model_validate(raw)

    def list(
        self,
        *,
        limit: Optional[int] = None,
        cursor: Optional[str] = None,
        include_disabled: Optional[bool] = None,
    ) -> AsyncPager[WebhookSubscriptionResponse]:
        return self._client.paginate(
            "/v1/webhook-subscriptions",
            item_key="subscriptions",
            params=_subscription_list_params(limit, cursor, include_disabled),
            item_factory=WebhookSubscriptionResponse.model_validate,
        )

    async def get(self, subscription_id: str) -> WebhookSubscriptionResponse:
        raw = await self._client.request(
            "GET", f"/v1/webhook-subscriptions/{subscription_id}"
        )
        return WebhookSubscriptionResponse.model_validate(raw)

    async def update(
        self, subscription_id: str, body: UpdateWebhookSubscriptionRequest
    ) -> WebhookSubscriptionResponse:
        raw = await self._client.request(
            "PATCH",
            f"/v1/webhook-subscriptions/{subscription_id}",
            json=_update_body(body),
        )
        return WebhookSubscriptionResponse.model_validate(raw)

    async def delete(self, subscription_id: str) -> None:
        await self._client.request_raw(
            "DELETE", f"/v1/webhook-subscriptions/{subscription_id}"
        )
        return None

    def list_deliveries(
        self,
        session_id: str,
        *,
        limit: Optional[int] = None,
        cursor: Optional[str] = None,
    ) -> AsyncPager[WebhookDeliveryResponse]:
        return self._client.paginate(
            f"/v1/sessions/{session_id}/webhook-deliveries",
            item_key="deliveries",
            params=_delivery_list_params(limit, cursor),
            item_factory=WebhookDeliveryResponse.model_validate,
        )

    async def list_latest_deliveries(
        self, session_id: str
    ) -> WebhookDeliveryListResponse:
        """The most recent delivery attempt per subscription; nothing is re-sent."""
        raw = await self._client.request(
            "GET", f"/v1/sessions/{session_id}/webhook-deliveries/latest"
        )
        return WebhookDeliveryListResponse.model_validate(raw)
