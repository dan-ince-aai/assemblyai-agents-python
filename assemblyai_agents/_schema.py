import inspect
from enum import Enum
from types import UnionType
from typing import (
    Any,
    Callable,
    Iterator,
    Literal,
    Mapping,
    Optional,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from pydantic import BaseModel, Field, create_model

from ._context import ToolContext
from ._exceptions import ConfigurationError

_PRIMITIVES = (str, int, float, bool)

_SUPPORTED = (
    "str, int, float, bool, list[T], dict[str, T], Literal[...], an Enum subclass, "
    "Optional[T], or a pydantic BaseModel subclass"
)

_UNUSABLE_KINDS = {
    inspect.Parameter.VAR_POSITIONAL: "*args",
    inspect.Parameter.VAR_KEYWORD: "**kwargs",
    inspect.Parameter.POSITIONAL_ONLY: "a positional-only parameter",
}

_REF_PREFIX = "#/$defs/"

# Keys the emitted schema does not carry. `title` is pydantic's echo of a class or
# field name, and `$defs` is pruned because every `$ref` is inlined at its use site
# instead. Both are matched against a *schema's own* keys only — see `_NAME_KEYED`.
_DROPPED_KEYS = ("$defs", "title")

# Keys whose value maps names to schemas rather than being a schema itself. Their
# keys come from whatever the customer called their parameters, so running the
# keyword filter over them drops a parameter named `title` while `required` goes on
# naming it — a schema the API refuses for a required field that is not present.
# `properties` is the only one pydantic emits for the annotations `_is_supported`
# admits: `additionalProperties` holds a single schema (the `dict[str, T]` value
# type), and `$defs`, the other name map, never reaches the walk because
# `_DROPPED_KEYS` prunes it unread.
_NAME_KEYED = ("properties",)

# A tool signature comes from whatever a customer wrote, and every guard here
# walks it recursively, so nesting depth needs an explicit bound. The cycle guard
# does not cover it: a chain of a thousand distinct nested models is perfectly
# acyclic. Nothing legitimate chains models this deep, and pydantic's own schema
# generation exhausts the stack somewhere past 55 levels, so the bound is set
# below that to keep the failure a clear error rather than a RecursionError.
_MAX_MODEL_DEPTH = 32

# Backstop on the two recursive walks over an emitted schema: `_inline`, which
# recurses about three levels per level of model nesting, and `_walk`, which
# recurses once per `$defs` level. `_MAX_MODEL_DEPTH` already keeps
# `derive_schema` well under this; the cap bounds both for any other shape that
# reaches them, so the failure is a clear error rather than a RecursionError.
_MAX_INLINE_DEPTH = 100


def derive_schema(
    func: Callable[..., Any],
    *,
    descriptions: Optional[Mapping[str, str]] = None,
) -> dict:
    tool_name = func.__name__
    hints = get_type_hints(func)
    fields = {}
    for name, parameter in inspect.signature(func).parameters.items():
        unusable = _UNUSABLE_KINDS.get(parameter.kind)
        if unusable is not None:
            raise ConfigurationError(
                f"tool `{tool_name}`, parameter `{name}`: {unusable} cannot be "
                f"described in a schema. The runtime calls the handler as "
                f"handler(**arguments), so every parameter must be a named keyword."
            )
        if name not in hints:
            raise ConfigurationError(
                f"tool `{tool_name}`, parameter `{name}`: no type hint. Every "
                f"parameter needs one, because the schema is derived from it."
            )
        annotation = hints[name]
        if annotation is ToolContext:
            continue
        _reject_unsupported(tool_name, name, annotation)
        # Ahead of `_reject_recursive`, which asks pydantic for a schema and would
        # hit its recursion limit first on an over-deep chain.
        _reject_too_deep(tool_name, name, annotation)
        _reject_recursive(tool_name, name, annotation)
        default = (
            ... if parameter.default is inspect.Parameter.empty else parameter.default
        )
        fields[name] = (
            annotation,
            Field(default, description=(descriptions or {}).get(name)),
        )

    raw = create_model(f"{tool_name}_parameters", **fields).model_json_schema()
    inlined = _inline(raw, raw.get("$defs", {}), tool_name)
    return {
        "type": inlined.get("type", "object"),
        "properties": inlined.get("properties", {}),
        # Pydantic omits `required` entirely when every parameter has a default, and
        # a root missing one of its three keys is a shape nothing downstream was
        # asked about.
        "required": inlined.get("required", []),
    }


def _inline(
    node: Any,
    defs: Mapping[str, Any],
    tool_name: str,
    path: tuple = (),
    depth: int = 0,
) -> Any:
    if depth > _MAX_INLINE_DEPTH:
        raise ConfigurationError(
            f"tool `{tool_name}`: the parameter schema nests deeper than "
            f"{_MAX_INLINE_DEPTH} levels at `{'.'.join(path)}`. Flatten the "
            f"nested models to a fixed depth."
        )
    if isinstance(node, list):
        return [
            _inline(item, defs, tool_name, path + (str(index),), depth + 1)
            for index, item in enumerate(node)
        ]
    if not isinstance(node, dict):
        return node
    reference = node.get("$ref")
    if reference is not None:
        resolved = _inline(
            defs[reference[len(_REF_PREFIX) :]], defs, tool_name, path, depth + 1
        )
        # A `$ref` carries siblings — a default and a description ride alongside it —
        # so they are merged over the definition instead of being dropped with it.
        siblings = {
            key: _inline_child(key, value, defs, tool_name, path, depth + 1)
            for key, value in node.items()
            if key not in ("$ref", "title")
        }
        return {**resolved, **siblings}
    return {
        key: _inline_child(key, value, defs, tool_name, path, depth + 1)
        for key, value in node.items()
        if key not in _DROPPED_KEYS
    }


def _inline_child(
    key: str,
    value: Any,
    defs: Mapping[str, Any],
    tool_name: str,
    path: tuple,
    depth: int,
) -> Any:
    if key in _NAME_KEYED and isinstance(value, dict):
        # Walked one level down, so the filter in `_inline` sees each property's own
        # schema body and never the names keying them.
        return {
            name: _inline(schema, defs, tool_name, path + (key, name), depth + 1)
            for name, schema in value.items()
        }
    return _inline(value, defs, tool_name, path + (key,), depth)


def _reject_unsupported(tool_name: str, parameter: str, annotation: Any) -> None:
    if _is_supported(annotation):
        return
    # Pydantic produces a schema for several of these rather than refusing them —
    # `Any` emits no `type`, a bare `list` emits `{"items": {}}` — so the annotation
    # is the only place they can be caught.
    raise ConfigurationError(
        f"tool `{tool_name}`, parameter `{parameter}`: type "
        f"`{_render(annotation)}` is not supported. Use one of: {_SUPPORTED}."
    )


def _is_supported(annotation: Any) -> bool:
    if annotation in _PRIMITIVES:
        return True
    if isinstance(annotation, type) and issubclass(annotation, (Enum, BaseModel)):
        return True
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is Literal:
        return True
    if origin in (Union, UnionType):
        optional = [arg for arg in args if arg is not type(None)]
        return len(args) == 2 and len(optional) == 1 and _is_supported(optional[0])
    if origin is list:
        return len(args) == 1 and _is_supported(args[0])
    if origin is dict:
        return len(args) == 2 and args[0] is str and _is_supported(args[1])
    return False


def _render(annotation: Any) -> str:
    if isinstance(annotation, type) and get_origin(annotation) is None:
        return annotation.__name__
    return str(annotation)


def _reject_recursive(tool_name: str, parameter: str, annotation: Any) -> None:
    for model in _models_in(annotation):
        cycle = _find_cycle(model.model_json_schema().get("$defs", {}), tool_name)
        if cycle is None:
            continue
        raise ConfigurationError(
            f"tool `{tool_name}`, parameter `{parameter}`: model "
            f"`{model.__name__}` is recursive ({' -> '.join(cycle)}). A recursive "
            f"schema cannot be inlined. Flatten it to a fixed depth."
        )


def _reject_too_deep(tool_name: str, parameter: str, annotation: Any) -> None:
    for model in _models_in(annotation):
        chain = _deepest_chain(model)
        if len(chain) <= _MAX_MODEL_DEPTH:
            continue
        rendered = " -> ".join(model.__name__ for model in chain[:4])
        raise ConfigurationError(
            f"tool `{tool_name}`, parameter `{parameter}`: models nest more than "
            f"{_MAX_MODEL_DEPTH} levels deep ({rendered} -> ...). "
            f"Flatten it to a fixed depth."
        )


def _deepest_chain(root: type) -> list:
    # Iterative, so measuring the depth cannot itself recurse. It stops as soon as
    # the limit is passed, so an absurdly deep chain costs no more than a real one.
    stack = [(root, [root])]
    deepest: list = []
    while stack:
        model, path = stack.pop()
        if len(path) > len(deepest):
            deepest = path
        if len(path) > _MAX_MODEL_DEPTH:
            return path
        for field in model.model_fields.values():
            for nested in _models_in(field.annotation):
                # Compared as classes, not by name: two distinct models can share a
                # name, and treating that as a cycle would undercount the depth. A
                # genuine repeat is a cycle, which `_reject_recursive` names.
                if nested not in path:
                    stack.append((nested, path + [nested]))
    return deepest


def _models_in(annotation: Any) -> Iterator[type]:
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        yield annotation
        return
    for arg in get_args(annotation):
        yield from _models_in(arg)


def _find_cycle(defs: Mapping[str, Any], tool_name: str) -> Optional[list]:
    # `cleared` spans the whole scan rather than one walk. Whether a cycle is
    # reachable from a definition does not depend on the path taken to reach it,
    # so a definition that reached none once will never reach one, and re-walking
    # it only costs time. Without it a `$defs` graph that shares definitions
    # across branches is re-traversed once per distinct path — exponential in the
    # number of levels.
    cleared: set = set()
    for name in defs:
        cycle = _walk(name, [], defs, tool_name, cleared)
        if cycle is not None:
            return cycle
    return None


def _walk(
    name: str,
    stack: list,
    defs: Mapping[str, Any],
    tool_name: str,
    cleared: set,
) -> Optional[list]:
    # `_reject_too_deep` runs first and keeps `derive_schema` well clear of this,
    # so it is a backstop for any other caller: the walk recurses once per `$defs`
    # level and would otherwise raise RecursionError instead of a usable error.
    if len(stack) > _MAX_INLINE_DEPTH:
        raise ConfigurationError(
            f"tool `{tool_name}`: the parameter schema chains more than "
            f"{_MAX_INLINE_DEPTH} model definitions deep at `{name}`. Flatten "
            f"the nested models to a fixed depth."
        )
    if name in stack:
        return stack[stack.index(name) :] + [name]
    # Only ever holds definitions whose whole reachable subgraph came back clean,
    # so skipping one cannot hide a cycle. A definition abandoned mid-walk because
    # a cycle was found is never recorded.
    if name in cleared:
        return None
    if name not in defs:
        return None
    for target in _refs_in(defs[name]):
        cycle = _walk(target, stack + [name], defs, tool_name, cleared)
        if cycle is not None:
            return cycle
    cleared.add(name)
    return None


def _refs_in(node: Any) -> Iterator[str]:
    if isinstance(node, list):
        for item in node:
            yield from _refs_in(item)
        return
    if not isinstance(node, dict):
        return
    for key, value in node.items():
        if key == "$ref" and isinstance(value, str):
            yield value[len(_REF_PREFIX) :]
        else:
            yield from _refs_in(value)
