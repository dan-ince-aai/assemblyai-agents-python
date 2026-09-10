import uuid
from typing import Any, Mapping, Optional

import httpx

from . import _retry
from ._config import ClientConfig
from ._exceptions import ResponseError, parse_error
from ._response import RawResponse
from ._version import __version__

_TERMINAL_KEY_REUSE = ("idempotency_key_reuse", 422)


def _user_agent() -> str:
    return f"assemblyai-agents-python/{__version__}"


def _build_headers(
    config: ClientConfig,
    json_body: Any,
    extra: Optional[Mapping[str, str]],
    idempotency_key: Optional[str],
) -> dict:
    headers: dict = {
        "Authorization": f"Bearer {config.api_key}",
        "User-Agent": _user_agent(),
    }
    if json_body is not None:
        headers["Content-Type"] = "application/json"
    if idempotency_key is not None:
        headers["Idempotency-Key"] = idempotency_key
    if extra:
        headers.update(extra)
    return headers


def _to_raw_response(response: httpx.Response) -> RawResponse:
    return RawResponse(
        status_code=response.status_code,
        headers=response.headers,
        content=response.content,
        request_id=response.headers.get("X-Request-Id"),
    )


def _decode_success(raw: RawResponse) -> Any:
    if not raw.content:
        return None
    try:
        return raw.json()
    except ValueError:
        raise ResponseError(raw)


def _envelope_code(response: httpx.Response) -> Optional[str]:
    try:
        body = response.json()
    except ValueError:
        return None
    if isinstance(body, dict):
        return body.get("code")
    return None


def _mint_idempotency_key(
    idempotent: bool, extra: Optional[Mapping[str, str]]
) -> Optional[str]:
    if not idempotent:
        return None
    # A caller-supplied Idempotency-Key disables auto-minting and is sent verbatim.
    if extra and any(k.lower() == "idempotency-key" for k in extra):
        return None
    # Minted once here, before the retry loop, and reused byte-for-byte on every
    # attempt — a fresh key per attempt would defeat server dedup and double-act.
    return uuid.uuid4().hex


class _Plan:
    def __init__(
        self,
        config: ClientConfig,
        method: str,
        path: str,
        params: Optional[Mapping[str, Any]],
        json_body: Any,
        headers: Optional[Mapping[str, str]],
        idempotent: bool,
        timeout: Optional[float],
    ) -> None:
        self.config = config
        self.method = method.upper()
        self.url = config.base_url + path
        self.params = params
        self.json_body = json_body
        self.idempotency_key = _mint_idempotency_key(idempotent, headers)
        self.headers = _build_headers(config, json_body, headers, self.idempotency_key)
        self.timeout = timeout if timeout is not None else config.timeout

    def build_request(self, http_client) -> httpx.Request:
        return http_client.build_request(
            self.method,
            self.url,
            params=self.params,
            json=self.json_body,
            headers=self.headers,
            timeout=httpx.Timeout(self.timeout),
        )


def _classify(response: httpx.Response, attempts_remaining: int) -> str:
    status = response.status_code
    if 200 <= status < 300:
        return "success"
    code = _envelope_code(response)
    if status == 422 and code == _TERMINAL_KEY_REUSE[0]:
        return "terminal"
    if _retry.is_retryable_status(status, code) and attempts_remaining > 0:
        return "retry"
    return "terminal"


class SyncTransportCore:
    def __init__(self, config: ClientConfig, http_client: httpx.Client) -> None:
        self._config = config
        self._http = http_client

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
        plan = _Plan(
            self._config, method, path, params, json, headers, idempotent, timeout
        )
        retry_index = 0
        attempts_remaining = self._config.max_retries
        while True:
            request = plan.build_request(self._http)
            try:
                response = self._http.send(request)
            except httpx.TransportError:
                if attempts_remaining <= 0:
                    raise
                _retry.sleep_sync(_retry.sleep_seconds(retry_index, None))
                retry_index += 1
                attempts_remaining -= 1
                continue
            outcome = _classify(response, attempts_remaining)
            if outcome == "retry":
                retry_after = response.headers.get("Retry-After")
                _retry.sleep_sync(_retry.sleep_seconds(retry_index, retry_after))
                retry_index += 1
                attempts_remaining -= 1
                continue
            return _to_raw_response(response)

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
        raw = self.request_raw(
            method,
            path,
            params=params,
            json=json,
            headers=headers,
            idempotent=idempotent,
            timeout=timeout,
        )
        if 200 <= raw.status_code < 300:
            return _decode_success(raw)
        raise parse_error(raw)


class AsyncTransportCore:
    def __init__(self, config: ClientConfig, http_client: httpx.AsyncClient) -> None:
        self._config = config
        self._http = http_client

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
        plan = _Plan(
            self._config, method, path, params, json, headers, idempotent, timeout
        )
        retry_index = 0
        attempts_remaining = self._config.max_retries
        while True:
            request = plan.build_request(self._http)
            try:
                response = await self._http.send(request)
            except httpx.TransportError:
                if attempts_remaining <= 0:
                    raise
                await _retry.sleep_async(_retry.sleep_seconds(retry_index, None))
                retry_index += 1
                attempts_remaining -= 1
                continue
            outcome = _classify(response, attempts_remaining)
            if outcome == "retry":
                retry_after = response.headers.get("Retry-After")
                await _retry.sleep_async(_retry.sleep_seconds(retry_index, retry_after))
                retry_index += 1
                attempts_remaining -= 1
                continue
            return _to_raw_response(response)

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
        raw = await self.request_raw(
            method,
            path,
            params=params,
            json=json,
            headers=headers,
            idempotent=idempotent,
            timeout=timeout,
        )
        if 200 <= raw.status_code < 300:
            return _decode_success(raw)
        raise parse_error(raw)
