"""Application service for loading provider-neutral tool definitions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .errors import OneCProtocolError
from .models import ToolDefinition, ToolDescriptor
from .onec_client import OneCClient

_MISSING = object()


class ToolRegistry:
    """Load dynamic 1C descriptors and validate their LLM-facing structure."""

    def __init__(self, client: OneCClient) -> None:
        self._client = client

    async def load_tools(self) -> list[ToolDefinition]:
        descriptors = await self._client.list_tools()
        definitions: list[ToolDefinition] = []
        for index, descriptor in enumerate(descriptors):
            self._validate_descriptor(descriptor, index=index)
            definitions.append(descriptor.as_definition())
        return definitions

    @staticmethod
    def _validate_descriptor(descriptor: Any, *, index: int) -> None:
        if not isinstance(descriptor, ToolDescriptor):
            raise OneCProtocolError(
                f"Invalid tool descriptor at index {index}: expected ToolDescriptor"
            )

        name = getattr(descriptor, "name", _MISSING)
        if not isinstance(name, str) or not name.strip():
            raise OneCProtocolError(
                f"Invalid tool descriptor at index {index}: name must be a non-empty string"
            )

        description = getattr(descriptor, "description", _MISSING)
        if not isinstance(description, str):
            raise OneCProtocolError(
                f"Invalid tool descriptor {name!r}: description must be a string"
            )

        read_only = getattr(descriptor, "read_only", _MISSING)
        if not isinstance(read_only, bool):
            raise OneCProtocolError(
                f"Invalid tool descriptor {name!r}: read_only must be a bool"
            )

        input_schema = getattr(descriptor, "input_schema", _MISSING)
        if not isinstance(input_schema, Mapping):
            raise OneCProtocolError(
                f"Invalid tool descriptor {name!r}: input_schema must be an object"
            )
        if input_schema.get("type") != "object":
            raise OneCProtocolError(
                f"Invalid tool descriptor {name!r}: input_schema.type must be 'object'"
            )
