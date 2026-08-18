"""Typed 1C API DTOs and minimal provider-neutral model types."""

from __future__ import annotations

from copy import deepcopy
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ToolDescriptor(ApiModel):
    name: str
    description: str
    read_only: bool = Field(strict=True)
    input_schema: dict[str, Any]

    def as_definition(self) -> ToolDefinition:
        """Represent the descriptor neutrally without changing its JSON Schema."""

        return ToolDefinition(
            name=self.name,
            description=self.description,
            read_only=self.read_only,
            parameters=deepcopy(self.input_schema),
        )


class ToolError(ApiModel):
    code: str
    message: str


class ToolResult(ApiModel):
    success: bool
    data: Any | None
    error: ToolError | None

    @model_validator(mode="after")
    def validate_result_invariants(self) -> ToolResult:
        if self.success and self.error is not None:
            raise ValueError("a successful tool result cannot contain an error")
        if not self.success and self.error is None:
            raise ValueError("a failed tool result must contain an error")
        return self


class HealthResponse(ApiModel):
    status: str
    api_version: str


class ToolsData(ApiModel):
    tools: list[ToolDescriptor]


class ToolsResponse(ApiModel):
    success: bool
    data: ToolsData | None
    error: ToolError | None

    @model_validator(mode="after")
    def validate_result_invariants(self) -> ToolsResponse:
        if self.success and self.error is not None:
            raise ValueError("a successful tools response cannot contain an error")
        if not self.success and self.error is None:
            raise ValueError("a failed tools response must contain an error")
        return self


class TransportErrorResponse(ApiModel):
    success: Literal[False]
    data: None
    error: ToolError


class ChatRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ChatMessage(ApiModel):
    role: ChatRole
    content: str


class ToolDefinition(ApiModel):
    name: str
    description: str
    parameters: dict[str, Any]
    read_only: bool = True


class ModelRequest(ApiModel):
    messages: list[ChatMessage]
    tools: list[ToolDefinition] = Field(default_factory=list)


class ModelResponse(ApiModel):
    message: ChatMessage
