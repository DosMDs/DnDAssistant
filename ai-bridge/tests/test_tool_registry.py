from __future__ import annotations

from typing import Any

import pytest

from dnd_ai_bridge.errors import OneCProtocolError
from dnd_ai_bridge.models import ToolDescriptor
from dnd_ai_bridge.ollama_tools import to_ollama_tools
from dnd_ai_bridge.tool_registry import ToolRegistry


class FakeOneCClient:
    def __init__(self, descriptors: list[Any]) -> None:
        self.descriptors = descriptors
        self.list_tools_calls = 0

    async def list_tools(self) -> list[Any]:
        self.list_tools_calls += 1
        return self.descriptors


@pytest.mark.asyncio
async def test_registry_loads_from_client_and_preserves_order_and_fields() -> None:
    first_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "futureKeyword": {"foo": "bar"},
            }
        },
        "required": ["query"],
        "additionalProperties": False,
        "x-future-schema-keyword": {"nested": [1, True, None]},
    }
    descriptors = [
        ToolDescriptor(
            name="second_from_1c",
            description="Первый в ответе",
            read_only=False,
            input_schema=first_schema,
        ),
        ToolDescriptor(
            name="first_alphabetically",
            description="Второй в ответе",
            read_only=True,
            input_schema={"type": "object", "properties": {}},
        ),
    ]
    client = FakeOneCClient(descriptors)
    registry = ToolRegistry(client)  # type: ignore[arg-type]

    tools = await registry.load_tools()

    assert client.list_tools_calls == 1
    assert [tool.name for tool in tools] == [
        "second_from_1c",
        "first_alphabetically",
    ]
    assert tools[0].description == "Первый в ответе"
    assert tools[0].read_only is False
    assert tools[0].parameters == first_schema


@pytest.mark.asyncio
async def test_registry_accepts_a_new_unknown_tool_without_code_changes() -> None:
    schema = {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": ["a", "b"],
            }
        },
        "required": ["mode"],
        "additionalProperties": False,
    }
    descriptor = ToolDescriptor(
        name="new_tool",
        description="Новый инструмент",
        read_only=True,
        input_schema=schema,
    )
    registry = ToolRegistry(FakeOneCClient([descriptor]))  # type: ignore[arg-type]

    tools = await registry.load_tools()

    assert len(tools) == 1
    assert tools[0].name == "new_tool"
    assert tools[0].parameters == schema
    assert to_ollama_tools(tools) == [
        {
            "type": "function",
            "function": {
                "name": "new_tool",
                "description": "Новый инструмент",
                "parameters": schema,
            },
        }
    ]


def invalid_descriptor(**overrides: Any) -> ToolDescriptor:
    values: dict[str, Any] = {
        "name": "valid_name",
        "description": "Описание",
        "read_only": True,
        "input_schema": {"type": "object"},
    }
    values.update(overrides)
    return ToolDescriptor.model_construct(**values)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "descriptor",
    [
        invalid_descriptor(name=""),
        invalid_descriptor(name="   "),
        invalid_descriptor(name=123),
        ToolDescriptor.model_construct(
            name="missing_description",
            read_only=True,
            input_schema={"type": "object"},
        ),
        invalid_descriptor(description=None),
        invalid_descriptor(read_only="true"),
        invalid_descriptor(input_schema=[]),
        invalid_descriptor(input_schema={"type": "array"}),
        invalid_descriptor(input_schema={}),
    ],
)
async def test_registry_rejects_structurally_invalid_descriptors(
    descriptor: ToolDescriptor,
) -> None:
    registry = ToolRegistry(FakeOneCClient([descriptor]))  # type: ignore[arg-type]

    with pytest.raises(OneCProtocolError, match="Invalid tool descriptor"):
        await registry.load_tools()


@pytest.mark.asyncio
async def test_registry_rejects_untyped_client_output_with_protocol_error() -> None:
    registry = ToolRegistry(FakeOneCClient([{"name": "raw"}]))  # type: ignore[arg-type]

    with pytest.raises(OneCProtocolError, match="expected ToolDescriptor"):
        await registry.load_tools()
