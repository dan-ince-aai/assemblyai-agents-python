"""Create the agent once, update it after that, and remember which one it is.

    agent_id = agent.deploy(public_url="https://agent.example.com", secret=SECRET)

The declaration is bound to the address first, so every tool URL, the
pre-connect handlers and the reply endpoint all point at that host. Then
either the stored id is updated — the whole declaration, because ``PUT``
replaces — or a new agent is created and its id written down.

Where the id lives, in order: the ``agent_id=`` argument, the ``AGENT_ID``
environment variable, then ``id_file`` (default ``.agent_id``, suffixed with the
host when it is not the default one, because ids do not cross hosts).
"""

import os
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlsplit

from ._client import Client
from ._config import DEFAULT_BASE_URL
from ._exceptions import ConfigurationError, NotFoundError

ENV_PUBLIC_URL = "PUBLIC_BASE_URL"
ENV_SECRET = "AGENT_SECRET"
ENV_AGENT_ID = "AGENT_ID"
DEFAULT_ID_FILE = ".agent_id"


def resolve_public_url(public_url: Optional[str], *, verb: str = "deploy") -> str:
    """The address the platform reaches this process at, or a clear error."""
    url = (public_url or os.environ.get(ENV_PUBLIC_URL, "")).rstrip("/")
    if url:
        return url
    raise ConfigurationError(
        f"no public address. The platform reaches this process over public HTTPS, and "
        f"it resolves every URL in DNS when the agent is created, so `{verb}()` needs "
        f"one now: pass `public_url=\"https://...\"` or set {ENV_PUBLIC_URL}. In "
        f"development that is a tunnel (ngrok, cloudflared); in production it is your "
        f"host. Nothing in the SDK starts a tunnel for you."
    )


def resolve_secret(secret: Optional[str]) -> Optional[str]:
    return secret or os.environ.get(ENV_SECRET) or None


def id_file_for(id_file: Optional[str], base_url: str) -> Path:
    path = Path(id_file or DEFAULT_ID_FILE)
    host = urlsplit(base_url).hostname or ""
    default_host = urlsplit(DEFAULT_BASE_URL).hostname or ""
    if host and host != default_host:
        # Agents live per host and ids do not cross, so a second host gets its
        # own note rather than overwriting the first.
        path = path.with_name(path.name + "." + host.split(".")[0])
    return path


def stored_agent_id(agent_id: Optional[str], id_file: Path) -> Optional[str]:
    if agent_id:
        return agent_id
    if os.environ.get(ENV_AGENT_ID):
        return os.environ[ENV_AGENT_ID]
    if id_file.exists():
        return id_file.read_text().strip() or None
    return None


def deploy(
    agent: Any,
    *,
    public_url: Optional[str] = None,
    secret: Optional[str] = None,
    client: Optional[Client] = None,
    agent_id: Optional[str] = None,
    id_file: Optional[str] = DEFAULT_ID_FILE,
    log=print,
) -> str:
    """Create or update the agent on the platform and return its id.

    ``public_url`` (or ``PUBLIC_BASE_URL``) is required when the declaration
    hosts anything — tools, a pre-connect handler, or a ``reply`` function.
    ``secret`` (or ``AGENT_SECRET``) becomes the bearer on every hosted route.
    """
    client = client or Client()
    bound = agent
    if agent.needs_address and not agent.public_url:
        bound = agent.hosted_at(resolve_public_url(public_url), secret=resolve_secret(secret))
    elif secret is not None and agent.secret is None:
        bound = agent.hosted_at(agent.public_url, secret=secret)

    path = id_file_for(id_file, client._config.base_url) if id_file else None
    existing = stored_agent_id(agent_id, path) if path else agent_id

    if existing:
        try:
            deployed = client.agents.update(existing, bound)
            if log:
                log(f"updated agent {deployed.id}")
            return deployed.id
        except NotFoundError:
            if log:
                log(f"agent {existing} is gone; creating a new one")

    deployed = client.agents.create(bound)
    if path:
        path.write_text(deployed.id)
    if log:
        log(f"created agent {deployed.id}")
    return deployed.id
