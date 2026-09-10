import pytest
from assemblyai_agents.models.rest import (
    BuiltinToolListResponse,
    BuiltinToolResponse,
)
from assemblyai_agents.resources.builtin_tools import (
    AsyncBuiltinToolsResource,
    BuiltinToolsResource,
)

from .conftest import Recorder

# Canned wire response mirroring the committed catalog's Luhn entry. The
# transport is mocked: these tests pin the SDK's request shape and typed
# parsing; the server-side catalog content is pinned by the server's own
# test suites.
_CATALOG_BODY = {
    "builtin_tools": [
        {
            "name": "aai_credit_card_luhn_check",
            "description": "Validate a credit-card number with the Luhn checksum.",
            "parameters": {
                "type": "object",
                "properties": {
                    "card_number": {
                        "type": "string",
                        "description": "The card number, digits only or spaced.",
                    }
                },
                "required": ["card_number"],
            },
            "execution_mode": "interactive",
        }
    ]
}


def test_sdk_builtin_tools_list(make_client, recorder: Recorder):
    client = make_client([_CATALOG_BODY], recorder)
    assert isinstance(client.builtin_tools, BuiltinToolsResource)

    result = client.builtin_tools.list()

    assert isinstance(result, BuiltinToolListResponse)
    (tool,) = result.builtin_tools
    assert isinstance(tool, BuiltinToolResponse)
    assert tool.name == "aai_credit_card_luhn_check"
    assert tool.description == _CATALOG_BODY["builtin_tools"][0]["description"]
    assert tool.parameters == _CATALOG_BODY["builtin_tools"][0]["parameters"]
    assert tool.execution_mode == "interactive"

    assert recorder.count == 1
    req = recorder.requests[0]
    assert req.method == "GET"
    assert req.url.path == "/v1/builtin-tools"
    assert dict(req.url.params) == {}
    assert req.content == b""
    assert recorder.header("Idempotency-Key") is None


@pytest.mark.asyncio
async def test_sdk_builtin_tools_list_async_parity(make_client, make_async_client):
    sync_rec, async_rec = Recorder(), Recorder()
    sync_client = make_client([_CATALOG_BODY], sync_rec)
    async_client = make_async_client([_CATALOG_BODY], async_rec)
    assert isinstance(async_client.builtin_tools, AsyncBuiltinToolsResource)
    assert not isinstance(sync_client.builtin_tools, AsyncBuiltinToolsResource)

    sync_result = sync_client.builtin_tools.list()
    async_result = await async_client.builtin_tools.list()

    assert isinstance(async_result, BuiltinToolListResponse)
    assert async_result.model_dump() == sync_result.model_dump()
    assert sync_rec.count == async_rec.count == 1
    for rec in (sync_rec, async_rec):
        req = rec.requests[0]
        assert req.method == "GET"
        assert req.url.path == "/v1/builtin-tools"
        assert req.headers.get("Idempotency-Key") is None


@pytest.mark.asyncio
async def test_builtin_tools_resources_construct_directly(
    make_client, make_async_client
):
    sync_rec, async_rec = Recorder(), Recorder()
    sync_resource = BuiltinToolsResource(make_client([_CATALOG_BODY], sync_rec))
    async_resource = AsyncBuiltinToolsResource(
        make_async_client([_CATALOG_BODY], async_rec)
    )

    sync_result = sync_resource.list()
    async_result = await async_resource.list()

    assert isinstance(sync_result, BuiltinToolListResponse)
    assert async_result.model_dump() == sync_result.model_dump()
    for rec in (sync_rec, async_rec):
        assert rec.count == 1
        assert rec.requests[0].method == "GET"
        assert rec.requests[0].url.path == "/v1/builtin-tools"
