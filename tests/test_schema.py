import enum
from typing import Any, Literal, Optional, Union

import pytest
from assemblyai_agents import ConfigurationError, _schema
from assemblyai_agents._context import ToolContext
from assemblyai_agents._schema import (
    _MAX_INLINE_DEPTH,
    _MAX_MODEL_DEPTH,
    _find_cycle,
    _inline,
    derive_schema,
)
from pydantic import BaseModel, create_model


class Size(str, enum.Enum):
    SMALL = "small"
    LARGE = "large"


class Address(BaseModel):
    street: str
    zip_code: Optional[str] = None


class Post(BaseModel):
    title: str
    body: str


class Tree(BaseModel):
    child: Optional["Tree"] = None


class Left(BaseModel):
    right: Optional["Right"] = None


class Right(BaseModel):
    left: Optional[Left] = None


Left.model_rebuild()


def test_primitives_and_required_order():
    def place_order(item: str, quantity: int, price: float, express: bool) -> dict:
        return {}

    schema = derive_schema(place_order)

    assert schema["properties"]["item"] == {"type": "string"}
    assert schema["properties"]["quantity"] == {"type": "integer"}
    assert schema["properties"]["price"] == {"type": "number"}
    assert schema["properties"]["express"] == {"type": "boolean"}
    assert schema["required"] == ["item", "quantity", "price", "express"]


def test_root_carries_only_the_three_keys():
    def lookup(order_id: str) -> dict:
        return {}

    schema = derive_schema(lookup)

    assert list(schema) == ["type", "properties", "required"]
    assert schema["type"] == "object"


def test_required_is_empty_list_when_every_parameter_has_a_default():
    def search(query: str = "", limit: int = 10) -> dict:
        return {}

    schema = derive_schema(search)

    # Pydantic omits the key entirely in this case.
    assert schema["required"] == []


def test_no_titles_and_no_defs_anywhere():
    def add_item(size: Size, where: Address, tags: list[Size]) -> dict:
        return {}

    schema = derive_schema(add_item)

    assert "title" not in _keys(schema)
    assert "$defs" not in _keys(schema)
    assert "$ref" not in _keys(schema)


# There is deliberately no companion test for a parameter named `$defs`: `$` is
# not legal in a Python identifier, so no signature can declare one.
def test_parameter_named_title_is_kept():
    def create_post(title: str, body: str) -> dict:
        return {}

    schema = derive_schema(create_post)

    # `properties` keys are parameter names, never schema keywords, so the keyword
    # filter must not reach them.
    assert list(schema["properties"]) == ["title", "body"]
    assert schema["properties"]["title"] == {"type": "string"}


def test_every_required_name_appears_in_properties():
    def create_post(title: str, body: str) -> dict:
        return {}

    schema = derive_schema(create_post)

    # The API refuses a schema whose `required` names a property that is absent, so
    # a dropped parameter reaches the customer as a deploy error about a missing
    # required field rather than as a missing parameter.
    assert schema["required"] == ["title", "body"]
    assert not set(schema["required"]) - set(schema["properties"])


def test_nested_model_field_named_title_is_kept():
    def publish(post: Post) -> dict:
        return {}

    schema = derive_schema(publish)

    post = schema["properties"]["post"]
    # The model's own `title` is stripped; its field called `title` is not. A filter
    # that cannot tell a schema from a map of names gets exactly one of these wrong.
    assert "title" not in post
    assert post["properties"]["title"] == {"type": "string"}
    assert post["required"] == ["title", "body"]


def test_field_named_title_is_kept_inside_a_list_item():
    def publish(posts: list[Post]) -> dict:
        return {}

    schema = derive_schema(publish)

    item = schema["properties"]["posts"]["items"]
    assert "title" not in item
    assert item["properties"]["title"] == {"type": "string"}
    assert item["required"] == ["title", "body"]


def test_field_named_title_is_kept_inside_a_dict_value():
    def publish(posts: dict[str, Post]) -> dict:
        return {}

    schema = derive_schema(publish)

    value = schema["properties"]["posts"]["additionalProperties"]
    # `additionalProperties` holds one schema rather than a map of names, so its own
    # `title` is still stripped while the field named `title` is kept.
    assert "title" not in value
    assert value["properties"]["title"] == {"type": "string"}


def test_enum_is_inlined():
    def pick(size: Size) -> dict:
        return {}

    schema = derive_schema(pick)

    assert schema["properties"]["size"] == {
        "enum": ["small", "large"],
        "type": "string",
    }


def test_nested_model_is_inlined():
    def deliver(where: Address) -> dict:
        return {}

    schema = derive_schema(deliver)

    where = schema["properties"]["where"]
    assert where["type"] == "object"
    assert where["required"] == ["street"]
    assert where["properties"]["street"] == {"type": "string"}


def test_ref_sibling_keys_are_merged_over_the_definition():
    def deliver(
        size: Size,
        where: Address = Address(street="1 Main St"),
    ) -> dict:
        return {}

    schema = derive_schema(
        deliver, descriptions={"size": "how big", "where": "where to send it"}
    )

    # Both fields emit `{"$ref": ..., ...siblings}`; an inliner that returns the
    # resolved definition alone drops the description and the default.
    size = schema["properties"]["size"]
    assert size["description"] == "how big"
    assert size["enum"] == ["small", "large"]

    where = schema["properties"]["where"]
    assert where["description"] == "where to send it"
    assert where["default"] == {"street": "1 Main St", "zip_code": None}
    assert where["type"] == "object"
    assert schema["required"] == ["size"]


def test_literal_inlines_as_enum_plus_type():
    def pick(size: Literal["small", "large"]) -> dict:
        return {}

    schema = derive_schema(pick)

    assert schema["properties"]["size"] == {
        "enum": ["small", "large"],
        "type": "string",
    }


def test_optional_keeps_its_anyof_and_stays_required():
    def note(text: Optional[str]) -> dict:
        return {}

    schema = derive_schema(note)

    text = schema["properties"]["text"]
    assert text == {"anyOf": [{"type": "string"}, {"type": "null"}]}
    # Optionality in the schema is independent of `required`: with no default the
    # caller still has to send the parameter.
    assert schema["required"] == ["text"]


def test_refs_are_chased_inside_items_and_additional_properties():
    def stock(places: list[Address], counts: dict[str, Size]) -> dict:
        return {}

    schema = derive_schema(stock)

    assert schema["properties"]["places"]["items"]["type"] == "object"
    assert schema["properties"]["counts"]["additionalProperties"]["enum"] == [
        "small",
        "large",
    ]


def test_refs_are_chased_inside_anyof():
    def deliver(where: Optional[Address]) -> dict:
        return {}

    schema = derive_schema(deliver)

    branches = schema["properties"]["where"]["anyOf"]
    assert branches[0]["properties"]["street"] == {"type": "string"}
    assert branches[1] == {"type": "null"}


def test_context_parameter_is_left_out_of_the_schema():
    def lookup(order_id: str, ctx: ToolContext) -> dict:
        return {}

    schema = derive_schema(lookup)

    assert list(schema["properties"]) == ["order_id"]
    assert schema["required"] == ["order_id"]


def test_self_referencing_model_is_rejected():
    def add_item(node: Tree) -> dict:
        return {}

    with pytest.raises(ConfigurationError) as exc_info:
        derive_schema(add_item)

    message = str(exc_info.value)
    assert "tool `add_item`, parameter `node`" in message
    assert "model `Tree` is recursive (Tree -> Tree)" in message
    assert "Flatten it to a fixed depth." in message


def test_mutually_recursive_models_are_rejected_with_the_cycle():
    def add_item(node: Left) -> dict:
        return {}

    with pytest.raises(ConfigurationError) as exc_info:
        derive_schema(add_item)

    message = str(exc_info.value)
    assert "model `Left` is recursive" in message
    assert "(Left -> Right -> Left)" in message


def _nested_model_chain(levels: int) -> type:
    model = create_model("ChainLeaf", value=(str, ...))
    for level in range(levels):
        model = create_model(f"ChainLevel{level}", child=(model, ...))
    return model


@pytest.mark.parametrize("levels", [60, 300])
def test_models_nested_past_the_depth_limit_are_refused(levels):
    def submit(payload) -> dict:
        return {}

    # Acyclic, so the cycle guard passes it through. Unguarded, this depth makes
    # pydantic's own schema generation raise RecursionError instead.
    submit.__annotations__["payload"] = _nested_model_chain(levels)

    with pytest.raises(ConfigurationError) as exc_info:
        derive_schema(submit)

    message = str(exc_info.value)
    assert "tool `submit`, parameter `payload`" in message
    assert "models nest more than 32 levels deep" in message
    assert "Flatten it to a fixed depth." in message


def test_same_named_distinct_models_still_count_toward_the_depth_limit():
    # Every level is a distinct class that happens to be named `Level`. Comparing
    # by name would read the second one as a cycle and let the chain through.
    model = create_model("Level", value=(str, ...))
    for _ in range(_MAX_MODEL_DEPTH + 5):
        model = create_model("Level", child=(model, ...))

    def submit(payload) -> dict:
        return {}

    submit.__annotations__["payload"] = model

    with pytest.raises(ConfigurationError) as exc_info:
        derive_schema(submit)

    assert "models nest more than" in str(exc_info.value)


def test_nesting_within_the_depth_limit_is_still_inlined():
    def submit(payload) -> dict:
        return {}

    submit.__annotations__["payload"] = _nested_model_chain(5)

    schema = derive_schema(submit)

    node = schema["properties"]["payload"]
    for _ in range(5):
        node = node["properties"]["child"]
    assert node["properties"]["value"] == {"type": "string"}


def test_recursive_model_reports_the_cycle_not_the_depth_limit():
    def add_item(node: Tree) -> dict:
        return {}

    with pytest.raises(ConfigurationError) as exc_info:
        derive_schema(add_item)

    message = str(exc_info.value)
    assert "model `Tree` is recursive" in message
    assert "nest more than" not in message


def test_inline_refuses_a_schema_past_the_traversal_limit():
    # The model-depth guard keeps `derive_schema` clear of this backstop, so the
    # walk is driven directly to prove it is bounded rather than stack-bound.
    node: Any = {"type": "string"}
    for _ in range(_MAX_INLINE_DEPTH + 2):
        node = {"properties": node}

    with pytest.raises(ConfigurationError) as exc_info:
        _inline(node, {}, "submit")

    message = str(exc_info.value)
    assert "tool `submit`" in message
    assert f"nests deeper than {_MAX_INLINE_DEPTH} levels" in message


def test_cycle_walk_refuses_a_definition_chain_past_the_traversal_limit():
    # `_reject_too_deep` keeps `derive_schema` clear of this backstop, so the
    # cycle walk is driven directly to prove it is bounded rather than
    # stack-bound. The chain is acyclic, so only the depth bound can stop it.
    defs = {
        f"Level{level}": {"$ref": f"#/$defs/Level{level + 1}"}
        for level in range(_MAX_INLINE_DEPTH + 2)
    }

    with pytest.raises(ConfigurationError) as exc_info:
        _find_cycle(defs, "submit")

    message = str(exc_info.value)
    assert "tool `submit`" in message
    assert f"more than {_MAX_INLINE_DEPTH}" in message
    assert "Flatten" in message


def test_cycle_walk_expands_each_shared_definition_once(monkeypatch):
    # A diamond graph: every level references the next twice. Without
    # visited-definition tracking the walk re-expands each shared definition
    # once per distinct path, which is 2**levels walks; with it, each
    # definition is expanded once. Acyclic, so the answer is None either way —
    # the tracking is what keeps getting there affordable.
    levels = 20
    defs: dict = {
        f"Level{level}": {
            "one": {"$ref": f"#/$defs/Level{level + 1}"},
            "two": {"$ref": f"#/$defs/Level{level + 1}"},
        }
        for level in range(levels)
    }
    defs[f"Level{levels}"] = {"type": "string"}

    calls = 0
    real_walk = _schema._walk

    def counting_walk(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_walk(*args, **kwargs)

    monkeypatch.setattr(_schema, "_walk", counting_walk)

    assert _find_cycle(defs, "submit") is None
    assert calls < 10 * len(defs)


def test_shared_definitions_do_not_hide_a_real_cycle():
    # Memoising "no cycle reachable from here" must not swallow a cycle that is
    # only reachable through a definition already expanded on another path.
    defs = {
        "Root": {
            "one": {"$ref": "#/$defs/Shared"},
            "two": {"$ref": "#/$defs/Loop"},
        },
        "Shared": {"type": "string"},
        "Loop": {"back": {"$ref": "#/$defs/Loop"}},
    }

    assert _find_cycle(defs, "submit") == ["Loop", "Loop"]


@pytest.mark.parametrize(
    "annotation,rendered",
    [
        (Any, "Any"),
        (list, "list"),
        (dict, "dict"),
        (tuple, "tuple"),
        (set, "set"),
        (tuple[int, str], "tuple[int, str]"),
        (set[int], "set[int]"),
        (dict[int, str], "dict[int, str]"),
        (Union[int, str], "typing.Union[int, str]"),
    ],
)
def test_unsupported_annotations_are_rejected(annotation, rendered):
    def wanted(thing) -> dict:
        return {}

    wanted.__annotations__["thing"] = annotation

    with pytest.raises(ConfigurationError) as exc_info:
        derive_schema(wanted)

    message = str(exc_info.value)
    assert f"tool `wanted`, parameter `thing`: type `{rendered}`" in message
    assert "is not supported" in message


def test_unannotated_parameter_is_rejected():
    def wanted(thing) -> dict:
        return {}

    with pytest.raises(ConfigurationError, match="no type hint"):
        derive_schema(wanted)


def test_varargs_and_positional_only_are_rejected():
    def with_varargs(item: str, *rest: str) -> dict:
        return {}

    def with_kwargs(item: str, **rest: str) -> dict:
        return {}

    def positional_only(item: str, /) -> dict:
        return {}

    with pytest.raises(ConfigurationError, match=r"\*args"):
        derive_schema(with_varargs)
    with pytest.raises(ConfigurationError, match=r"\*\*kwargs"):
        derive_schema(with_kwargs)
    with pytest.raises(ConfigurationError, match="positional-only"):
        derive_schema(positional_only)


def test_string_annotations_are_resolved():
    def pick(size: "Size") -> dict:
        return {}

    assert derive_schema(pick)["properties"]["size"]["enum"] == ["small", "large"]


def _keys(node) -> set:
    found = set()
    if isinstance(node, list):
        for item in node:
            found |= _keys(item)
    elif isinstance(node, dict):
        found |= set(node)
        for value in node.values():
            found |= _keys(value)
    return found
