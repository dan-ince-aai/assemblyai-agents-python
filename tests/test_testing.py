import pytest
from assemblyai_agents import ConfigurationError, ToolContext, VoiceAgent, tool
from assemblyai_agents import testing as testing_module
from assemblyai_agents.testing import create_tool_context, get_tool

pytestmark = [pytest.mark.asyncio]

ORDERS_URL = "https://api.pizzapalace.com/orders/W004"


def declare() -> VoiceAgent:
    @tool
    async def lookup_order(order_id: str, ctx: ToolContext) -> dict:
        """Look up one of the caller's orders by its ID.

        Args:
            order_id: The order number the caller read out.
        """
        response = await ctx.http.get(
            f"https://api.pizzapalace.com/orders/{order_id}",
            headers={"authorization": ctx.secret("orders_api_key")},
        )
        if response.status_code == 404:
            raise LookupError(f"no order {order_id}")
        return response.json()

    return VoiceAgent(
        name="Pizza Line",
        voice="ivy",
        system_prompt="Take orders.",
        tools=[lookup_order],
    )


async def test_a_stubbed_request_is_served_and_recorded():
    agent = declare()
    ctx = create_tool_context(secrets={"orders_api_key": "test-key"})
    ctx.http.stub("GET", ORDERS_URL, json={"status": "shipped"})

    lookup_order = get_tool(agent, "lookup_order")

    assert await lookup_order(order_id="W004", ctx=ctx) == {"status": "shipped"}
    assert ctx.http.calls[0].headers["authorization"] == "test-key"
    assert ctx.http.calls[0].method == "GET"


async def test_a_stubbed_error_status_comes_back_rather_than_raising():
    agent = declare()
    ctx = create_tool_context(secrets={"orders_api_key": "test-key"})
    ctx.http.stub("GET", ORDERS_URL, status_code=404)

    lookup_order = get_tool(agent, "lookup_order")

    with pytest.raises(LookupError):
        await lookup_order(order_id="W004", ctx=ctx)


async def test_an_unstubbed_request_is_refused_and_names_the_url():
    ctx = create_tool_context()

    with pytest.raises(ConfigurationError) as exc_info:
        await ctx.http.get(ORDERS_URL)

    message = str(exc_info.value)
    assert f"GET {ORDERS_URL} is not stubbed" in message
    assert "nothing is stubbed yet" in message
    # Refused, but still recorded, so a test can see what the tool reached for.
    assert ctx.http.calls[0].url == ORDERS_URL


async def test_an_unstubbed_request_lists_what_is_stubbed():
    ctx = create_tool_context()
    ctx.http.stub("POST", "https://api.pizzapalace.com/orders")

    with pytest.raises(ConfigurationError) as exc_info:
        await ctx.http.get(ORDERS_URL)

    assert "POST https://api.pizzapalace.com/orders" in str(exc_info.value)


async def test_an_unset_secret_raises_and_names_itself():
    ctx = create_tool_context(secrets={"other_key": "x"})

    with pytest.raises(ConfigurationError) as exc_info:
        ctx.secret("orders_api_key")

    message = str(exc_info.value)
    assert "secret `orders_api_key` is not set" in message
    assert "`other_key`" in message


async def test_a_set_secret_is_returned():
    ctx = create_tool_context(secrets={"orders_api_key": "test-key"})

    assert ctx.secret("orders_api_key") == "test-key"


async def test_session_ids_auto_increment():
    first = create_tool_context()
    second = create_tool_context()

    assert first.session_id != second.session_id


async def test_aborted_is_a_settable_bool():
    ctx = create_tool_context()

    assert ctx.aborted is False
    ctx.aborted = True
    assert ctx.aborted is True


async def test_the_log_captures_what_a_tool_writes():
    ctx = create_tool_context()

    ctx.log.info("looked_up", order_id="W004")

    assert ctx.log.records[0].level == "info"
    assert ctx.log.records[0].event == "looked_up"
    assert ctx.log.records[0].fields == {"order_id": "W004"}


async def test_the_double_satisfies_the_protocol():
    assert isinstance(create_tool_context(), ToolContext)


async def test_get_tool_names_the_type_it_was_handed():
    declare()

    with pytest.raises(ConfigurationError) as exc_info:
        get_tool("Pizza Line", "lookup_order")

    message = str(exc_info.value)
    assert "wants the VoiceAgent declaration and got str" in message


async def test_get_tool_says_to_import_the_module_when_nothing_is_declared():
    agent = VoiceAgent(name="Pizza Line", voice="ivy", system_prompt="Take orders.")

    with pytest.raises(ConfigurationError) as exc_info:
        get_tool(agent, "lookup_order")

    message = str(exc_info.value)
    assert "lists no tools" in message
    assert "VoiceAgent(tools=[...])" in message


async def test_get_tool_names_what_is_declared_when_the_name_is_wrong():
    agent = declare()

    with pytest.raises(ConfigurationError) as exc_info:
        get_tool(agent, "lookup_orders")

    message = str(exc_info.value)
    assert "lists no tool under `lookup_orders`" in message
    assert "`lookup_order`" in message


async def test_nothing_here_opens_a_connection():
    # The double exists so a tool can be tested with no network at all; an httpx
    # import here would be the first step back toward one.
    assert not hasattr(testing_module, "httpx")
