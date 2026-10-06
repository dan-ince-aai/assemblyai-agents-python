import re
from urllib.parse import urlparse

from ._exceptions import ConfigurationError

# Hostnames AssemblyAI mints for the service it hosts for an agent. Mirrors the
# pattern the API enforces, so an address of this shape is refused here instead
# of coming back as a 422 on the agent write.
#
# The published address is `<workspace>-<environment>--svc-<32 hex>.modal.run`
# and only the label after `--` is AssemblyAI's to choose: the hex is a digest
# of the deployment ID, which is why one of these addresses names one deployment
# and stops answering when the next supersedes it. Matching the label alone
# recognises every region and environment without naming any of them, and
# anchoring at `.modal.run` keeps a customer's own domain out of it however much
# it resembles one of ours.
_HOSTS_ASSEMBLYAI_MINTS = (
    re.compile(r"--svc-[0-9a-f]{32}\.modal\.run$", re.IGNORECASE),
)

# The API's ceiling on a stored address, path form included.
MAX_URL_LENGTH = 2048


def is_hosted_service_path(url: str) -> bool:
    """Whether this is a route on the service AssemblyAI hosts for the agent.

    A stored address has always needed a hostname to be reachable, so a value
    with none can only be a path meant for the hosted service.
    """
    return url.startswith("/")


def names_a_hosted_service_address(url: str) -> bool:
    """Whether this absolute URL points at one deployment's minted address.

    Matched on the host alone. The path belongs to the customer and says nothing
    about who owns the name in front of it.
    """
    host = urlparse(url).hostname
    if not host:
        return False
    return any(pattern.search(host) for pattern in _HOSTS_ASSEMBLYAI_MINTS)


def check_called_address(url: str, *, subject: str, allow_http: bool = True) -> None:
    """Refuse an address the API would refuse, naming the form that works.

    `subject` prefixes every message so the caller reads which declaration is at
    fault; it ends in whatever separator that caller already uses.
    """
    if is_hosted_service_path(url):
        _check_hosted_service_path(url, subject=subject)
        return
    schemes = ("https://", "http://") if allow_http else ("https://",)
    if not url.startswith(schemes):
        raise ConfigurationError(
            f"{subject}`{url}` is neither a path on the service AssemblyAI hosts "
            f"for this agent, which starts with `/`, nor "
            f"{'an http(s)' if allow_http else 'an https'} URL of your own. The "
            f"platform fetches this address itself, so it has to be one of the two."
        )
    if names_a_hosted_service_address(url):
        raise ConfigurationError(
            f"{subject}`{url}` is one deployment of the service AssemblyAI hosts "
            f"for this agent, and stops answering when that agent is deployed "
            f"again, so the API refuses it. Write the path on its own — "
            f"`{_path_of(url)}` — and AssemblyAI resolves it to whichever "
            f"deployment is live when the call arrives."
        )
    _check_length(url, subject=subject)


def _check_hosted_service_path(path: str, *, subject: str) -> None:
    _check_length(path, subject=subject)
    if any(character.isspace() for character in path):
        raise ConfigurationError(
            f"{subject}`{path}` contains whitespace. A path on the service "
            f"AssemblyAI hosts is stored as written."
        )
    parsed = urlparse(path)
    if parsed.scheme or parsed.netloc:
        raise ConfigurationError(
            f"{subject}`{path}` starts with `/` but carries a scheme or a host. A "
            f"path on the service AssemblyAI hosts carries neither; write an "
            f"absolute https:// URL to call somewhere else."
        )


def _check_length(url: str, *, subject: str) -> None:
    if len(url) > MAX_URL_LENGTH:
        raise ConfigurationError(
            f"{subject}the address is {len(url)} characters. The API stores at "
            f"most {MAX_URL_LENGTH}."
        )


def _path_of(url: str) -> str:
    """The replacement to hand back, taken from the address they already wrote."""
    return urlparse(url).path or "/"
