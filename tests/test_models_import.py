import enum
import importlib
import sys
import typing
from pathlib import Path

from assemblyai_agents.models import rest, ws


def _enum_member_count(annotation: object) -> int:
    seen: list[type] = []

    def walk(tp: object) -> None:
        if isinstance(tp, type) and issubclass(tp, enum.Enum):
            seen.append(tp)
            return
        for arg in typing.get_args(tp):
            walk(arg)

    walk(annotation)
    if not seen:
        raise AssertionError(f"no Enum found in annotation {annotation!r}")
    return len(list(seen[0]))


def test_rest_models_import_and_key_types():
    assert hasattr(rest, "AgentListResponse")
    assert hasattr(rest, "ErrorResponse")

    list_fields = set(rest.AgentListResponse.model_fields)
    assert {"agents", "response_metadata"} <= list_fields

    error_fields = set(rest.ErrorResponse.model_fields)
    assert {"code", "message", "param", "request_id"} <= error_fields


def test_ws_models_import_and_key_types():
    assert hasattr(ws, "SessionConfig")
    assert hasattr(ws, "SessionError")

    code_field = ws.SessionError.model_fields["code"]
    assert _enum_member_count(code_field.annotation) == 16


def test_no_future_import_in_generated_models():
    assert "from __future__" not in Path(rest.__file__).read_text()
    assert "from __future__" not in Path(ws.__file__).read_text()


def test_codegen_is_build_time_only_not_in_library():
    importlib.import_module("assemblyai_agents")
    importlib.import_module("assemblyai_agents.models.rest")
    importlib.import_module("assemblyai_agents.models.ws")

    assert "datamodel_code_generator" not in sys.modules

    for module in (rest, ws):
        source = Path(module.__file__).read_text()
        assert "datamodel_code_generator" not in source
        assert "import datamodel" not in source
