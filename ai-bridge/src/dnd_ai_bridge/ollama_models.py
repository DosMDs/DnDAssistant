"""Typed DTOs for the local Ollama HTTP API boundary."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class OllamaApiModel(BaseModel):
    """Allow additive Ollama response fields while typing the fields we use."""

    model_config = ConfigDict(extra="allow")


class OllamaToolFunction(OllamaApiModel):
    name: str
    arguments: dict[str, Any]
    description: str | None = None
    index: int | None = None


class OllamaToolCall(OllamaApiModel):
    function: OllamaToolFunction
    type: str | None = None


class OllamaChatMessage(OllamaApiModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    thinking: str | None = None
    tool_calls: list[OllamaToolCall] = Field(default_factory=list)
    tool_name: str | None = None


class OllamaChatRequest(OllamaApiModel):
    model: str
    messages: list[OllamaChatMessage]
    tools: list[dict[str, Any]] = Field(default_factory=list)
    stream: bool = False
    think: bool = False
    options: dict[str, Any] | None = None
    keep_alive: str | int | None = None


class OllamaChatResponse(OllamaApiModel):
    model: str
    created_at: str
    message: OllamaChatMessage
    done: bool
    done_reason: str | None = None
    total_duration: int | None = Field(default=None, ge=0)
    load_duration: int | None = Field(default=None, ge=0)
    prompt_eval_count: int | None = Field(default=None, ge=0)
    prompt_eval_duration: int | None = Field(default=None, ge=0)
    eval_count: int | None = Field(default=None, ge=0)
    eval_duration: int | None = Field(default=None, ge=0)


class OllamaVersionResponse(OllamaApiModel):
    version: str


class OllamaModelDetails(OllamaApiModel):
    parent_model: str | None = None
    format: str | None = None
    family: str | None = None
    families: list[str] | None = None
    parameter_size: str | None = None
    quantization_level: str | None = None


class OllamaModelSummary(OllamaApiModel):
    name: str
    model: str | None = None
    modified_at: str | None = None
    size: int | None = Field(default=None, ge=0)
    digest: str | None = None
    details: OllamaModelDetails | None = None


class OllamaModelsResponse(OllamaApiModel):
    models: list[OllamaModelSummary]


class OllamaShowModelResponse(OllamaApiModel):
    name: str | None = None
    model: str | None = None
    digest: str | None = None
    parameters: str | None = None
    license: str | None = None
    modified_at: str | None = None
    details: OllamaModelDetails | None = None
    template: str | None = None
    capabilities: list[str] = Field(default_factory=list)
    model_info: dict[str, Any] = Field(default_factory=dict)


class OllamaRunningModel(OllamaApiModel):
    """A model currently resident in memory, as reported by ``/api/ps``."""

    name: str | None = None
    model: str | None = None
    digest: str | None = None
    size: int | None = Field(default=None, ge=0)
    size_vram: int | None = Field(default=None, ge=0)
    context_length: int | None = Field(default=None, ge=0)
    expires_at: str | None = None
    details: OllamaModelDetails | None = None

    @model_validator(mode="after")
    def require_identifier(self) -> OllamaRunningModel:
        if not (self.name or self.model):
            raise ValueError("a running model must include name or model")
        return self

    @property
    def identifier(self) -> str:
        return self.model or self.name or ""


class OllamaRunningModelsResponse(OllamaApiModel):
    models: list[OllamaRunningModel]


class OllamaUnloadResponse(OllamaApiModel):
    """Minimal response contract for an explicit keep_alive=0 unload."""

    model: str
    done: bool


class OllamaErrorResponse(OllamaApiModel):
    error: str
