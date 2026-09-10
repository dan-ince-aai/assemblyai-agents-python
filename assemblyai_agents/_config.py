import os
from dataclasses import dataclass, field

from ._exceptions import ConfigurationError

ENV_API_KEY = "ASSEMBLYAI_API_KEY"
DEFAULT_BASE_URL = "https://agents.assemblyai.com"
DEFAULT_TIMEOUT = 30.0
DEFAULT_MAX_RETRIES = 3


def _resolve_api_key(api_key: str | None) -> str:
    if api_key:
        return api_key
    env_value = os.environ.get(ENV_API_KEY)
    if env_value:
        return env_value
    raise ConfigurationError(
        f"No API key provided. Pass api_key= or set {ENV_API_KEY}."
    )


@dataclass
class ClientConfig:
    api_key: str = field(repr=False)
    base_url: str
    timeout: float
    max_retries: int

    @classmethod
    def build(
        cls,
        api_key: str | None,
        base_url: str,
        timeout: float,
        max_retries: int,
    ) -> "ClientConfig":
        return cls(
            api_key=_resolve_api_key(api_key),
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            max_retries=max_retries,
        )
