import pytest
from assemblyai_agents import Client, ConfigurationError

from .conftest import Recorder, make_sync_transport


def test_auth_header_from_env(monkeypatch, recorder: Recorder):
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "k-env")
    transport = make_sync_transport([(200, {"ok": True}, None)], recorder)
    client = Client(transport=transport, base_url="https://agents.test.local")

    client.request("GET", "/v1/agents")

    assert recorder.header("Authorization") == "Bearer k-env"


def test_auth_header_arg_overrides_env(monkeypatch, recorder: Recorder):
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "k-env")
    transport = make_sync_transport([(200, {"ok": True}, None)], recorder)
    client = Client(
        api_key="k-arg", transport=transport, base_url="https://agents.test.local"
    )

    client.request("GET", "/v1/agents")

    assert recorder.header("Authorization") == "Bearer k-arg"


def test_missing_api_key_raises_configuration_error(monkeypatch):
    monkeypatch.delenv("ASSEMBLYAI_API_KEY", raising=False)
    with pytest.raises(ConfigurationError):
        Client(base_url="https://agents.test.local")


def test_empty_env_api_key_treated_as_unset(monkeypatch):
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "")
    with pytest.raises(ConfigurationError):
        Client(base_url="https://agents.test.local")


def test_api_key_never_in_exception_or_repr(monkeypatch, recorder: Recorder):
    secret = "k-super-secret-9f3a"
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", secret)
    # A domain error so the SDK builds an exception that references the request.
    transport = make_sync_transport(
        [(404, {"code": "agent_not_found", "request_id": "req-1"}, None)], recorder
    )
    client = Client(
        api_key=secret, transport=transport, base_url="https://agents.test.local"
    )

    from assemblyai_agents import NotFoundError

    with pytest.raises(NotFoundError) as exc_info:
        client.request("GET", "/v1/agents/nope")

    exc = exc_info.value
    assert secret not in str(exc)
    assert secret not in repr(exc)
    # The exception carries the raw response; its stringification must not leak.
    assert secret not in repr(exc.raw)
    assert secret not in str(exc.raw)
    # The client's own repr must not embed the key either.
    assert secret not in repr(client)
    # Nor the config dataclass the client holds (its api_key field is repr=False).
    assert secret not in repr(client._config)


def test_api_key_not_in_raw_response_or_logs(monkeypatch, caplog, recorder: Recorder):
    secret = "k-super-secret-leak-probe"
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", secret)
    transport = make_sync_transport([(200, {"ok": True}, None)], recorder)
    client = Client(
        api_key=secret, transport=transport, base_url="https://agents.test.local"
    )

    with caplog.at_level("DEBUG", logger="assemblyai_agents"):
        raw = client.request_raw("GET", "/v1/agents")

    # The request DID carry the Bearer key (proves the test isn't a false pass).
    assert recorder.header("Authorization") == f"Bearer {secret}"
    # But the RawResponse the caller receives never surfaces it: RawResponse
    # exposes the RESPONSE headers, not the request Authorization.
    assert secret not in str(dict(raw.headers))
    assert all(secret not in str(v) for v in raw.headers.values())
    assert secret not in raw.content.decode(errors="ignore")
    # Nothing the SDK logged contains the key.
    assert secret not in caplog.text
