from typing import Dict, Optional, Type

from ._response import RawResponse


class ErrorCode:
    """The `code` values the API can return, as plain `str` constants.

    Deliberately NOT a StrEnum and NOT a Literal alias. A newer server can emit
    a code this SDK version has never heard of, and that must still compare
    cleanly as a string: a Literal makes every comparison a type error, and a
    StrEnum invites `ErrorCode(e.code)`, which raises on an unknown code —
    breaking the exact forward-compatibility case the status-keyed hierarchy
    exists to serve. Compare with `e.code == ErrorCode.AGENT_NOT_FOUND`; an
    unrecognised code simply compares False.

    This list is a hand-maintained copy of the service's error catalog. Nothing
    machine-readable links the two, so it drifts silently when the server adds
    a code.
    """

    MISSING_AUTHORIZATION = "missing_authorization"
    UNAUTHORIZED = "unauthorized"
    AUTH_SERVICE_UNAVAILABLE = "auth_service_unavailable"
    VALIDATION_ERROR = "validation_error"
    INVALID_REQUEST = "invalid_request"
    AGENT_NOT_FOUND = "agent_not_found"
    PHONE_NUMBER_NOT_FOUND = "phone_number_not_found"
    CALL_NOT_FOUND = "call_not_found"
    SESSION_NOT_FOUND = "session_not_found"
    WEBHOOK_SUBSCRIPTION_NOT_FOUND = "webhook_subscription_not_found"
    WEBHOOK_DELIVERIES_NOT_FOUND = "webhook_deliveries_not_found"
    PHONE_NUMBER_CONFLICT = "phone_number_conflict"
    WEBHOOK_SUBSCRIPTION_ALREADY_EXISTS = "webhook_subscription_already_exists"
    PHONE_NUMBER_NOT_AVAILABLE = "phone_number_not_available"
    PHONE_NUMBER_HAS_NO_AGENT = "phone_number_has_no_agent"
    TELEPHONY_PROVIDER_ERROR = "telephony_provider_error"
    IDEMPOTENCY_KEY_INVALID = "idempotency_key_invalid"
    IDEMPOTENCY_KEY_REUSE = "idempotency_key_reuse"
    IDEMPOTENCY_IN_PROGRESS = "idempotency_in_progress"
    NOT_FOUND = "not_found"
    METHOD_NOT_ALLOWED = "method_not_allowed"
    INTERNAL_ERROR = "internal_error"


class AssemblyAIAgentsError(Exception):
    pass


class ConfigurationError(AssemblyAIAgentsError):
    pass


class RealtimeError(AssemblyAIAgentsError):
    def __init__(self, message: str, *, close_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.close_code = close_code


class WebhookVerificationError(AssemblyAIAgentsError):
    pass


class WebhookSignatureError(WebhookVerificationError):
    pass


class WebhookTimestampError(WebhookVerificationError):
    pass


class APIError(AssemblyAIAgentsError):
    def __init__(
        self,
        *,
        status: int,
        code: Optional[str],
        message: str,
        param: Optional[str],
        request_id: Optional[str],
        errors: Optional[list],
        raw: RawResponse,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.param = param
        self.request_id = request_id
        self.errors = errors
        self.raw = raw

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(status={self.status!r}, code={self.code!r}, "
            f"message={self.message!r}, request_id={self.request_id!r})"
        )


class BadRequestError(APIError):
    """400 and 405: the request itself was malformed or used the wrong method."""


class AuthenticationError(APIError):
    """401: the API key was missing, malformed, or rejected."""


class NotFoundError(APIError):
    """404: the addressed resource does not exist."""


class ConflictError(APIError):
    """409: the request collides with the current state of the resource."""


class ValidationError(APIError):
    """422: the request was well-formed but its contents were rejected."""


class ServerError(APIError):
    """500, 502 and 503: the request failed inside the service or a dependency."""


class ResponseError(APIError):
    """A 2xx response whose body is not valid JSON.

    An APIError subclass despite the success status, so that a caller who
    catches APIError to handle "the call did not produce a usable result" also
    catches this. `status` is the 2xx the server actually sent and `code` is
    None, because a success response carries no error envelope.
    """

    def __init__(self, raw: RawResponse) -> None:
        super().__init__(
            status=raw.status_code,
            code=None,
            message=(
                f"the server returned HTTP {raw.status_code}, a success status, but "
                f"the response body is not valid JSON, so the SDK has no result to "
                f"return. This is the service violating its own response contract, "
                f"not a problem with your request, so changing the request will not "
                f"help. Retry the call; if it keeps happening, read the undecoded "
                f"body from this exception's `.raw.content` and report request_id "
                f"{raw.request_id!r} to AssemblyAI support."
            ),
            param=None,
            request_id=raw.request_id,
            errors=None,
            raw=raw,
        )


# Status -> exception class. Keyed on the HTTP status, not the error code, so a
# code this SDK has never heard of still lands on the right class. Anything not
# listed here degrades to the base APIError rather than guessing from the
# status family.
_STATUS_REGISTRY: Dict[int, Type[APIError]] = {
    400: BadRequestError,
    401: AuthenticationError,
    404: NotFoundError,
    405: BadRequestError,
    409: ConflictError,
    422: ValidationError,
    500: ServerError,
    502: ServerError,
    503: ServerError,
}


def _decode_envelope(raw: RawResponse) -> Optional[dict]:
    # `None` distinguishes a body that could not be read as a JSON object at all
    # from one that read fine and merely omitted a field. The two are different
    # failures and `parse_error` reports them differently; an empty `{}` envelope
    # is the latter.
    try:
        body = raw.json()
    except ValueError:
        return None
    if not isinstance(body, dict):
        return None
    return body


def _unreadable_body_message(raw: RawResponse, request_id: Optional[str]) -> str:
    return (
        f"the server returned HTTP {raw.status_code} with no readable error "
        f"envelope, so no error code or reason is available. The body was empty, "
        f"was not JSON, or was not a JSON object, which usually means an edge "
        f"proxy or load balancer answered instead of the API. Read the raw body "
        f"from this exception's `.raw.content` to see what replied, and report "
        f"request_id {request_id!r} to AssemblyAI support if it persists."
    )


def _envelope_without_message(raw: RawResponse, request_id: Optional[str]) -> str:
    return (
        f"the server returned HTTP {raw.status_code} with a JSON error envelope "
        f"carrying no `message`, so the API stated no reason for the failure. "
        f"This is a gap in the response, not a malformed body. Check this "
        f"exception's `.code` and `.errors` for the machine-readable detail, and "
        f"report request_id {request_id!r} to AssemblyAI support if those do not "
        f"explain it."
    )


def parse_error(raw: RawResponse) -> APIError:
    decoded = _decode_envelope(raw)
    envelope = decoded if decoded is not None else {}
    code = envelope.get("code")
    param = envelope.get("param")
    errors = envelope.get("errors")
    # request_id lives in the envelope on a well-formed error, but an edge/proxy
    # 5xx has no parseable body — fall back to the X-Request-Id response header.
    request_id = envelope.get("request_id") or raw.request_id
    message = envelope.get("message")
    if not message:
        # Which fallback fires is keyed on whether the body decoded, not on
        # whether `message` was there: a decoded envelope that simply omits the
        # field must not be told its body was unreadable.
        message = (
            _envelope_without_message(raw, request_id)
            if decoded is not None
            else _unreadable_body_message(raw, request_id)
        )

    exc_class = _STATUS_REGISTRY.get(raw.status_code, APIError)
    return exc_class(
        status=raw.status_code,
        code=code,
        message=message,
        param=param,
        request_id=request_id,
        errors=errors,
        raw=raw,
    )
