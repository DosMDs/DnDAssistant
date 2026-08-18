from __future__ import annotations

import pytest
from pydantic import ValidationError

from dnd_ai_bridge.models import (
    ChatMessage,
    ChatRole,
    ModelRequest,
    ModelResponse,
    ToolDescriptor,
    ToolError,
    ToolResult,
)


def test_tool_result_success_contract() -> None:
    result = ToolResult(success=True, data={"x": 1}, error=None)
    assert result.data == {"x": 1}


def test_tool_result_failure_contract() -> None:
    result = ToolResult(
        success=False,
        data=None,
        error=ToolError(code="invalid_arguments", message="Неверные аргументы"),
    )
    assert result.error is not None
    assert result.error.code == "invalid_arguments"


@pytest.mark.parametrize(
    "payload",
    [
        {"success": True, "data": None, "error": {"code": "bad", "message": "bad"}},
        {"success": False, "data": None, "error": None},
    ],
)
def test_tool_result_rejects_broken_invariants(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ToolResult.model_validate(payload)


def test_descriptor_to_definition_preserves_schema_without_transformation() -> None:
    schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "title": "Строка поиска"},
            "limit": {"type": "integer", "minimum": 1},
        },
        "required": ["query"],
        "additionalProperties": False,
        "x-onec-extension": {"НеизвестноеПоле": [1, True, None]},
    }
    descriptor = ToolDescriptor(
        name="search_entities",
        description="Ищет сущности",
        read_only=True,
        input_schema=schema,
    )

    definition = descriptor.as_definition()

    assert definition.parameters == schema
    assert definition.read_only is True


def test_unicode_round_trip_in_typed_models() -> None:
    phrases = [
        "Торвальд Железнорукий",
        "5 Миртула 1492 ЛД",
        "Гильдия кузнецов",
    ]
    result = ToolResult(success=True, data={"values": phrases}, error=None)

    encoded = result.model_dump_json()
    decoded = ToolResult.model_validate_json(encoded)

    assert decoded.data == {"values": phrases}


def test_minimal_provider_neutral_models() -> None:
    request = ModelRequest(
        messages=[ChatMessage(role=ChatRole.USER, content="Где Торвальд?")]
    )
    response = ModelResponse(
        message=ChatMessage(role=ChatRole.ASSISTANT, content="Пока неизвестно.")
    )

    assert request.tools == []
    assert request.messages[0].role == ChatRole.USER
    assert response.message.role == ChatRole.ASSISTANT
