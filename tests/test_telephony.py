import pytest
from assemblyai_agents import (
    Captured,
    ConfigurationError,
    Header,
    HumanTransfer,
    PreConnectRequest,
    VoiceAgent,
)
from assemblyai_agents.models.rest import (
    HttpMethod,
    HttpToolHeaderInput,
    PlaintextHttpToolConfig,
    PlaintextPreConnectRequest,
    PreConnectReturn,
    TransferTarget,
)

PROMPT = "You take pizza orders."
TRUNK = "ST_abc123"


def agent(**kwargs) -> VoiceAgent:
    return VoiceAgent(name="Pizza Line", voice="ivy", system_prompt=PROMPT, **kwargs)


def whois(**kwargs) -> PreConnectRequest:
    kwargs.setdefault("url", "https://example.com/whois")
    return PreConnectRequest(**kwargs)


def test_a_human_transfer_becomes_the_wire_target():
    target = HumanTransfer(name="manager", phone_number="+14155550123")

    assert target.to_target() == TransferTarget(
        name="manager", kind="human", phone_number="+14155550123", mode="cold"
    )


def test_a_warm_transfer_carries_the_consult_fields():
    target = HumanTransfer(
        name="manager",
        phone_number="+14155550123",
        mode="warm",
        ring_timeout=30,
        consult_instructions="Say the caller is asking about a refund.",
        consult_timeout=20,
        record_consult=True,
    )

    assert target.to_target() == TransferTarget(
        name="manager",
        kind="human",
        phone_number="+14155550123",
        mode="warm",
        ring_timeout_seconds=30,
        consult_instructions="Say the caller is asking about a refund.",
        consult_timeout_seconds=20,
        record_consult=True,
    )


def test_the_transfer_targets_reach_the_request():
    built = agent(
        transfer_targets=[HumanTransfer(name="manager", phone_number="+14155550123")],
        outbound_trunk_id=TRUNK,
    ).to_request()

    assert built.transfer_targets == [
        TransferTarget(
            name="manager", kind="human", phone_number="+14155550123", mode="cold"
        )
    ]
    assert built.outbound_trunk_id == TRUNK


@pytest.mark.parametrize("number", ["4155550123", "+0155550123", "+1-415-555-0123", ""])
def test_a_transfer_number_that_is_not_e164_is_refused(number):
    with pytest.raises(ConfigurationError) as exc_info:
        HumanTransfer(name="manager", phone_number=number)

    assert "E.164" in str(exc_info.value)


@pytest.mark.parametrize("seconds", [0, 601])
def test_a_ring_timeout_outside_the_range_is_refused(seconds):
    with pytest.raises(ConfigurationError) as exc_info:
        HumanTransfer(name="manager", phone_number="+14155550123", ring_timeout=seconds)

    assert "1-600" in str(exc_info.value)


@pytest.mark.parametrize(
    "field", ["consult_instructions", "consult_timeout", "record_consult"]
)
def test_a_consult_field_on_a_cold_transfer_is_refused(field):
    # On a cold transfer the server silently ignores the first two and rejects
    # the third; refusing all three here means the SDK does not reproduce that split.
    values = {
        "consult_instructions": "Explain the refund.",
        "consult_timeout": 20,
        "record_consult": True,
    }

    with pytest.raises(ConfigurationError) as exc_info:
        HumanTransfer(
            name="manager",
            phone_number="+14155550123",
            mode="cold",
            **{field: values[field]},
        )

    assert field in str(exc_info.value)


def test_an_unknown_transfer_mode_is_refused():
    with pytest.raises(ConfigurationError) as exc_info:
        HumanTransfer(name="manager", phone_number="+14155550123", mode="hot")

    assert "mode" in str(exc_info.value)


def test_a_transfer_target_without_a_trunk_is_refused():
    with pytest.raises(ConfigurationError) as exc_info:
        agent(
            transfer_targets=[
                HumanTransfer(name="manager", phone_number="+14155550123")
            ]
        )

    assert "outbound_trunk_id" in str(exc_info.value)


def test_a_caller_id_that_is_not_e164_is_refused():
    with pytest.raises(ConfigurationError) as exc_info:
        agent(caller_id="4155550123")

    assert "E.164" in str(exc_info.value)


def test_a_caller_id_reaches_the_request():
    assert agent(caller_id="+14155550123").to_request().caller_id == "+14155550123"


def test_a_pre_connect_request_becomes_the_wire_model():
    entry = PreConnectRequest(
        url="https://example.com/whois",
        headers=[Header(name="Authorization", value="Bearer x")],
        returns=[Captured(name="customer_tier", path="customer.tier", default="std")],
        timeout_ms=250,
        allow_overrides=True,
    )

    assert entry.to_request() == PlaintextPreConnectRequest(
        http=PlaintextHttpToolConfig(
            url="https://example.com/whois",
            http_method=HttpMethod.POST,
            headers=[HttpToolHeaderInput(name="Authorization", value="Bearer x")],
        ),
        returns=[
            PreConnectReturn(name="customer_tier", path="customer.tier", default="std")
        ],
        timeout_ms=250,
        allow_overrides=["greeting"],
    )


def test_the_pre_connect_entries_reach_the_request():
    built = agent(pre_connect=[whois()]).to_request()

    assert built.pre_connect_requests == [whois().to_request()]


def test_a_header_needs_an_explicit_value():
    # Stored header secrets are re-joined to an incoming list BY POSITION on
    # the server, so a name-only keep is index-joined and two entries both named `Authorization` swap
    # credentials on reorder. Requiring a value removes index-joining entirely.
    with pytest.raises(TypeError):
        Header(name="Authorization")


def test_three_pre_connect_entries_are_refused():
    with pytest.raises(ConfigurationError) as exc_info:
        agent(pre_connect=[whois(), whois(), whois()])

    assert "at most 2" in str(exc_info.value)


def test_a_url_that_is_not_https_is_refused():
    with pytest.raises(ConfigurationError) as exc_info:
        PreConnectRequest(url="http://example.com/whois")

    assert "https" in str(exc_info.value)


@pytest.mark.parametrize("timeout", [0, 801])
def test_a_pre_connect_timeout_outside_the_range_is_refused(timeout):
    with pytest.raises(ConfigurationError) as exc_info:
        whois(timeout_ms=timeout)

    assert "1-800" in str(exc_info.value)


def test_allow_overrides_is_a_flag_not_a_list():
    with pytest.raises(ConfigurationError) as exc_info:
        whois(allow_overrides=["greeting"])

    assert "allow_overrides" in str(exc_info.value)


def test_a_sends_name_no_entry_produces_is_refused():
    with pytest.raises(ConfigurationError) as exc_info:
        agent(pre_connect=[whois(sends=["customer_tier"])])

    assert "customer_tier" in str(exc_info.value)


def test_a_sends_name_produced_by_a_later_entry_is_refused():
    first = whois(sends=["customer_tier"])
    second = PreConnectRequest(
        url="https://example.com/tier",
        returns=[Captured(name="customer_tier", path="customer.tier")],
    )

    with pytest.raises(ConfigurationError) as exc_info:
        agent(pre_connect=[first, second])

    assert "earlier" in str(exc_info.value)


def test_a_sends_name_produced_by_an_earlier_entry_is_accepted():
    first = PreConnectRequest(
        url="https://example.com/tier",
        returns=[Captured(name="customer_tier", path="customer.tier")],
    )
    second = whois(sends=["customer_tier"])

    built = agent(pre_connect=[first, second]).to_request()

    assert built.pre_connect_requests[1].sends == ["customer_tier"]


def test_a_capture_name_reused_across_entries_is_refused():
    first = PreConnectRequest(
        url="https://example.com/tier",
        returns=[Captured(name="customer_tier", path="customer.tier")],
    )
    second = PreConnectRequest(
        url="https://example.com/whois",
        returns=[Captured(name="customer_tier", path="whois.tier")],
    )

    with pytest.raises(ConfigurationError) as exc_info:
        agent(pre_connect=[first, second])

    assert "customer_tier" in str(exc_info.value)


def test_a_capture_name_reused_inside_one_entry_is_refused():
    with pytest.raises(ConfigurationError) as exc_info:
        whois(
            returns=[
                Captured(name="customer_tier", path="customer.tier"),
                Captured(name="customer_tier", path="whois.tier"),
            ]
        )

    assert "customer_tier" in str(exc_info.value)
