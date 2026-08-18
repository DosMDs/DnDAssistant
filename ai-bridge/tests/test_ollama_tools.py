from __future__ import annotations

from copy import deepcopy

from dnd_ai_bridge.models import ToolDefinition
from dnd_ai_bridge.ollama_tools import to_ollama_tools


def test_maps_tool_definition_to_exact_ollama_function_descriptor() -> None:
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "minLength": 1},
        },
        "required": ["query"],
        "additionalProperties": False,
    }
    tool = ToolDefinition(
        name="search_entities",
        description="Ищет сущности",
        read_only=True,
        parameters=parameters,
    )

    result = to_ollama_tools([tool])

    assert result == [
        {
            "type": "function",
            "function": {
                "name": "search_entities",
                "description": "Ищет сущности",
                "parameters": parameters,
            },
        }
    ]
    assert "read_only" not in result[0]
    assert "read_only" not in result[0]["function"]


def test_mapper_preserves_unknown_schema_keywords_and_has_no_mutation_side_effects() -> None:
    parameters = {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": ["a", "b"],
                "futureKeyword": {"foo": "bar"},
            }
        },
        "x-new-keyword": {"values": [1, 2]},
    }
    tool = ToolDefinition(
        name="new_tool",
        description="Новый инструмент",
        read_only=False,
        parameters=parameters,
    )
    before = deepcopy(tool.parameters)

    result = to_ollama_tools((tool,))
    result[0]["function"]["parameters"]["x-new-keyword"]["values"].append(3)

    assert tool.parameters == before
