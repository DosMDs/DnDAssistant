"""Infrastructure mapping to Ollama native function-tool descriptors."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from typing import Any

from .models import ToolDefinition


def to_ollama_tools(tools: Sequence[ToolDefinition]) -> list[dict[str, Any]]:
    """Map neutral tools to Ollama payloads without altering their JSON Schema."""

    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": deepcopy(tool.parameters),
            },
        }
        for tool in tools
    ]
