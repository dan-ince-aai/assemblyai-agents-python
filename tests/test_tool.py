import threading

import pytest
from assemblyai_agents import ConfigurationError, Tool, ToolContext, tool
from assemblyai_agents._tool import DEFAULT_TIMEOUT_SECONDS, ToolSpec
from assemblyai_agents.models.rest import (
    DtmfCollectionProfile,
    ExecutionMode,
    HttpMethod,
    HttpToolHeaderInput,
    PlaintextHttpToolConfig,
    ResponseInstructions,
)

pytestmark = [pytest.mark.asyncio]


def spec_of(func) -> ToolSpec:
    return func.spec


async def test_a_sync_tool_is_declared_and_stays_callable():
    @tool
    def lookup_order(order_id: str) -> dict:
        """Look up one of the caller's orders by its ID."""
        return {"order_id": order_id}

    assert isinstance(lookup_order, Tool)
    assert lookup_order.name == "lookup_order"
    assert lookup_order(order_id="W004") == {"order_id": "W004"}
    assert spec_of(lookup_order).is_async is False


async def test_an_async_tool_is_declared_and_stays_awaitable():
    @tool
    async def lookup_order(order_id: str) -> dict:
        """Look up one of the caller's orders by its ID."""
        return {"order_id": order_id}

    assert await lookup_order(order_id="W004") == {"order_id": "W004"}
    assert spec_of(lookup_order).is_async is True


async def test_a_sync_tool_is_invoked_on_a_worker_thread():
    @tool
    def where_am_i() -> dict:
        """Report the thread the handler ran on."""
        return {"thread": threading.get_ident()}

    result = await spec_of(where_am_i).invoke()

    # The event loop is carrying the call's audio, so a sync handler must not
    # hold it for the length of its own timeout.
    assert result["thread"] != threading.get_ident()


async def test_the_context_is_detected_by_annotation_and_left_out_of_the_schema():
    @tool
    async def lookup_order(order_id: str, ctx: ToolContext) -> dict:
        """Look up one of the caller's orders by its ID."""
        return {"session": ctx.session_id}

    spec = spec_of(lookup_order)
    assert spec.context_parameter == "ctx"
    assert list(spec.parameters["properties"]) == ["order_id"]
    assert spec.parameters["required"] == ["order_id"]


async def test_a_parameter_named_ctx_of_another_type_is_not_injected():
    @tool
    async def lookup_order(ctx: str) -> dict:
        """Look up an order by the context string the caller read out."""
        return {"ctx": ctx}

    spec = spec_of(lookup_order)
    # Injection is opt-in by annotation: a handler is called as
    # handler(**arguments), so injecting on the name would break this tool.
    assert spec.context_parameter is None
    assert list(spec.parameters["properties"]) == ["ctx"]


async def test_the_context_is_passed_under_the_name_the_signature_uses():
    seen = {}

    @tool
    async def lookup_order(order_id: str, call: ToolContext) -> dict:
        """Look up one of the caller's orders by its ID."""
        seen["session"] = call.session_id
        return {}

    class Context:
        session_id = "sess_9"

    await spec_of(lookup_order).invoke(context=Context(), order_id="W004")

    assert seen["session"] == "sess_9"


async def test_the_description_is_the_docstrings_first_paragraph():
    @tool
    async def lookup_order(order_id: str) -> dict:
        """Look up one of the caller's orders by its ID.

        Any second paragraph is for whoever reads the file, not for the model.

        Args:
            order_id: The order number the caller read out.
        """
        return {}

    assert spec_of(lookup_order).description == (
        "Look up one of the caller's orders by its ID."
    )


async def test_the_args_block_lands_in_the_schema():
    @tool
    async def place_order(size: str, quantity: int) -> dict:
        """Place a pizza order.

        Args:
            size: The size the caller asked for.
            quantity: How many, as a whole number that
                may wrap onto a second line.

        Returns:
            The order that was placed.
        """
        return {}

    properties = spec_of(place_order).parameters["properties"]
    assert properties["size"]["description"] == "The size the caller asked for."
    assert properties["quantity"]["description"] == (
        "How many, as a whole number that may wrap onto a second line."
    )


async def test_a_tool_with_no_docstring_is_refused():
    with pytest.raises(ConfigurationError) as exc_info:

        @tool
        async def lookup_order(order_id: str) -> dict:
            return {}

    assert "no description" in str(exc_info.value)


async def test_the_timeout_defaults_to_the_rows_own_default():
    @tool
    async def lookup_order(order_id: str) -> dict:
        """Look up one of the caller's orders by its ID."""
        return {}

    assert spec_of(lookup_order).timeout_seconds == DEFAULT_TIMEOUT_SECONDS == 120


@pytest.mark.parametrize("seconds", [1, 8, 300])
async def test_the_timeout_bounds_are_inclusive(seconds):
    @tool(timeout_seconds=seconds)
    async def lookup_order(order_id: str) -> dict:
        """Look up one of the caller's orders by its ID."""
        return {}

    assert spec_of(lookup_order).timeout_seconds == seconds


@pytest.mark.parametrize("seconds", [0, -1, 301, 3600])
async def test_a_timeout_outside_the_bounds_is_refused(seconds):
    with pytest.raises(ConfigurationError) as exc_info:

        @tool(timeout_seconds=seconds)
        async def lookup_order(order_id: str) -> dict:
            """Look up one of the caller's orders by its ID."""
            return {}

    assert "1-300 seconds" in str(exc_info.value)


async def test_the_decorator_arguments_are_the_generated_rest_types():
    @tool(
        timeout_seconds=8,
        execution_mode=ExecutionMode.interactive,
        response_instructions=ResponseInstructions(success="Mention the ETA."),
        http=PlaintextHttpToolConfig(
            url="https://api.pizzapalace.com/orders",
            http_method=HttpMethod.POST,
            headers=[HttpToolHeaderInput(name="Authorization", value="Bearer x")],
        ),
        dtmf_collected_arguments=[
            DtmfCollectionProfile(
                parameter_name="card_number",
                min_digits=15,
                max_digits=16,
                prompt="Enter your card number, then press pound.",
            )
        ],
    )
    async def take_payment(card_number: str) -> dict:
        """Take a card payment for the order."""
        return {}

    spec = spec_of(take_payment)
    assert spec.timeout_seconds == 8
    assert spec.execution_mode is ExecutionMode.interactive
    assert spec.response_instructions.success == "Mention the ETA."
    assert spec.http.headers[0].name == "Authorization"
    assert spec.dtmf_collected_arguments[0].parameter_name == "card_number"


async def test_execution_mode_hold_is_not_in_v1():
    with pytest.raises(ConfigurationError) as exc_info:

        @tool(execution_mode=ExecutionMode.hold)
        async def lookup_order(order_id: str) -> dict:
            """Look up one of the caller's orders by its ID."""
            return {}

    assert "not in v1" in str(exc_info.value)


async def test_the_tool_takes_no_name_argument():
    with pytest.raises(ConfigurationError) as exc_info:
        tool("lookup_order")

    assert "@tool takes no name" in str(exc_info.value)


async def test_the_definition_is_the_wire_model_the_agent_sends():
    @tool(timeout_seconds=8)
    async def lookup_order(order_id: str) -> dict:
        """Look up one of the caller's orders by its ID.

        Args:
            order_id: The order number the caller read out.
        """
        return {}

    definition = lookup_order.definition()

    assert definition.name == "lookup_order"
    assert definition.description == "Look up one of the caller's orders by its ID."
    assert definition.timeout_seconds == 8
    assert definition.parameters == {
        "type": "object",
        "properties": {
            "order_id": {
                "type": "string",
                "description": "The order number the caller read out.",
            }
        },
        "required": ["order_id"],
    }


async def test_an_undeclared_execution_mode_is_left_out_of_the_payload():
    # The stored agent always holds an explicit mode — the server fills in its
    # default at write time — so absent and explicit `interactive` are identical
    # once stored, and a future server-side default change would not move
    # stored agents.
    @tool
    async def lookup_order(order_id: str) -> dict:
        """Look up one of the caller's orders by its ID."""
        return {}

    definition = lookup_order.definition()

    assert definition.execution_mode is None
    payload = definition.model_dump(mode="json", exclude_none=True, by_alias=True)
    assert "execution_mode" not in payload


async def test_a_declared_execution_mode_is_sent():
    @tool(execution_mode=ExecutionMode.interactive)
    async def lookup_order(order_id: str) -> dict:
        """Look up one of the caller's orders by its ID."""
        return {}

    payload = lookup_order.definition().model_dump(mode="json", exclude_none=True)
    assert payload["execution_mode"] == "interactive"


@pytest.mark.parametrize(
    "name", ["aai_credit_card_luhn_check", "aai_pre_connect_context"]
)
async def test_a_name_that_collides_with_a_platform_tool_is_refused(name):
    # The server selects a tool's backend by NAME first, so a tool of this
    # name would route to the platform implementation and never reach the
    # customer's own code.
    async def handler(order_id: str) -> dict:
        """Look up one of the caller's orders by its ID."""
        return {}

    handler.__name__ = name

    with pytest.raises(ConfigurationError) as exc_info:
        tool(handler)

    message = str(exc_info.value)
    assert name in message
    assert "platform tool" in message


async def test_a_tool_with_no_return_annotation_is_refused():
    with pytest.raises(ConfigurationError) as exc_info:

        @tool
        async def lookup_order(order_id: str):
            """Look up one of the caller's orders by its ID."""
            return {}

    assert "no return annotation" in str(exc_info.value)


async def test_a_tool_whose_return_type_is_not_serialisable_is_refused():
    with pytest.raises(ConfigurationError) as exc_info:

        @tool
        async def lookup_order(order_id: str) -> object:
            """Look up one of the caller's orders by its ID."""
            return object()

    assert "not JSON-serialisable" in str(exc_info.value)


async def test_an_unresolvable_annotation_is_refused_by_name():
    with pytest.raises(ConfigurationError) as exc_info:

        @tool
        async def lookup_order(order_id: "NoSuchTypeAnywhere") -> dict:  # noqa: F821
            """Look up one of the caller's orders by its ID."""
            return {}

    assert "does not resolve" in str(exc_info.value)


async def test_a_name_that_is_not_snake_case_is_refused():
    with pytest.raises(ConfigurationError) as exc_info:

        @tool
        async def LookupOrder(order_id: str) -> dict:  # noqa: N802
            """Look up one of the caller's orders by its ID."""
            return {}

    assert "snake_case" in str(exc_info.value)


async def test_varargs_cannot_be_described_in_a_schema():
    with pytest.raises(ConfigurationError) as exc_info:

        @tool
        async def lookup_order(*order_ids: str) -> dict:
            """Look up one of the caller's orders by its ID."""
            return {}

    assert "*args" in str(exc_info.value)


async def test_a_parameter_with_no_type_hint_is_refused():
    with pytest.raises(ConfigurationError) as exc_info:

        @tool
        async def lookup_order(order_id) -> dict:
            """Look up one of the caller's orders by its ID."""
            return {}

    assert "no type hint" in str(exc_info.value)


# --------------------------------------------------------------------------- hosted_at


def _shelf():
    @tool(timeout_seconds=9)
    def check_stock(item: str, aisle: int = 1) -> dict:
        """Check whether an item is on the shelf.

        Args:
            item: What the caller asked for.
            aisle: Where to look first.
        """
        return {"in_stock": 3}

    return check_stock


def test_hosted_at_binds_an_address_the_decorator_could_not_know():
    # A tunnel address does not exist at import, which is when `@tool` runs, so
    # this is the supported way to point a declared tool at one.
    bound = _shelf().hosted_at("https://demo.ngrok-free.app/tools/check_stock")
    assert bound.definition().http.url == "https://demo.ngrok-free.app/tools/check_stock"
    assert bound.definition().http.http_method == HttpMethod.POST


def test_hosted_at_leaves_the_original_alone():
    # The module-level tool stays importable by tests, and one run's tunnel
    # cannot leak into a declaration built later in the same process.
    declared = _shelf()
    declared.hosted_at("https://demo.ngrok-free.app/tools/check_stock")
    assert declared.spec.http is None


def test_hosted_at_keeps_everything_the_model_reads():
    declared = _shelf()
    bound = declared.hosted_at("https://demo.ngrok-free.app/tools/check_stock")
    assert bound.name == declared.name
    assert bound.spec.description == declared.spec.description
    assert bound.spec.parameters == declared.spec.parameters
    assert bound.spec.timeout_seconds == 9
    assert bound.spec.target is declared.spec.target


def test_hosted_at_carries_headers_and_method():
    bound = _shelf().hosted_at(
        "https://demo.ngrok-free.app/stock",
        http_method=HttpMethod.GET,
        headers=[HttpToolHeaderInput(name="Authorization", value="Bearer s")],
    )
    http = bound.definition().http
    assert http.http_method == HttpMethod.GET
    assert [(h.name, h.value) for h in http.headers] == [("Authorization", "Bearer s")]


def test_hosted_at_refuses_a_relative_url():
    # The platform fetches the address itself, so a path alone is unreachable
    # and would only surface as the model apologising to a caller.
    with pytest.raises(ConfigurationError, match="not an http"):
        _shelf().hosted_at("/tools/check_stock")


async def test_a_bound_tool_still_runs():
    bound = _shelf().hosted_at("https://demo.ngrok-free.app/tools/check_stock")
    assert await bound.invoke(item="drill") == {"in_stock": 3}


# --------------------------------------------------------------------------- hosted here, or served elsewhere


def test_a_bare_tool_is_hosted_by_this_process():
    assert _shelf().hosted is True
    assert _shelf().spec.http is None


def test_url_points_the_platform_at_a_service_you_already_run():
    @tool(url="https://api.example.com/weather")
    def weather(city: str) -> dict:
        """Weather for a city."""
        return {}

    assert weather.hosted is False
    assert weather.spec.http.url == "https://api.example.com/weather"
    assert weather.spec.http.http_method == HttpMethod.POST


def test_url_takes_a_method_and_headers():
    @tool(url="https://api.example.com/weather", http_method=HttpMethod.GET,
          headers=[HttpToolHeaderInput(name="X-Key", value="k")])
    def weather(city: str) -> dict:
        """Weather for a city."""
        return {}

    assert weather.spec.http.http_method == HttpMethod.GET
    assert weather.spec.http.headers[0].name == "X-Key"


def test_a_url_that_is_not_http_is_refused():
    with pytest.raises(ConfigurationError, match="not an http\\(s\\) URL"):
        @tool(url="/tools/weather")
        def weather(city: str) -> dict:
            """Weather for a city."""
            return {}


def test_method_or_headers_without_a_url_is_refused():
    with pytest.raises(ConfigurationError, match="only make sense with url="):
        @tool(http_method=HttpMethod.GET)
        def weather(city: str) -> dict:
            """Weather for a city."""
            return {}


def test_http_is_the_deprecated_spelling_of_url():
    with pytest.warns(DeprecationWarning, match="http=.*deprecated"):
        @tool(http=PlaintextHttpToolConfig(url="https://api.example.com/weather", http_method=HttpMethod.POST))
        def weather(city: str) -> dict:
            """Weather for a city."""
            return {}

    assert weather.hosted is False
    assert weather.spec.http.url == "https://api.example.com/weather"
