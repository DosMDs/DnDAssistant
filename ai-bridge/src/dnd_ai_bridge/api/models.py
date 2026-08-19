"""Public HTTP request and response DTOs."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from ..models import ChatMessage


class HttpModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HealthResponse(HttpModel):
    status: str = "ok"


class AgentRunRequest(HttpModel):
    messages: list[ChatMessage] = Field(min_length=1)


class AgentRunResponse(HttpModel):
    response: str
    request_id: str


class ErrorDetail(HttpModel):
    code: str
    message: str


class ErrorResponse(HttpModel):
    error: ErrorDetail
    request_id: str
