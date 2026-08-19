"""Provider adapter from neutral model requests to local Ollama chat calls."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Callable

from pydantic import BaseModel, ConfigDict, Field

from .models import (
    ChatMessage,
    ChatRole,
    ModelPerformanceMetrics,
    ModelRequest,
    ModelResponse,
    ModelStreamChunk,
    ModelToolCall,
    ModelUsageMetrics,
)
from .ollama_client import OllamaClient
from .ollama_models import (
    OllamaChatMessage,
    OllamaChatRequest,
    OllamaChatResponse,
    OllamaToolCall,
)
from .ollama_tools import to_ollama_tools


class OllamaGenerationSettings(BaseModel):
    """Provider-specific generation settings; never leak into ModelRequest."""

    model_config = ConfigDict(extra="forbid")

    temperature: float | None = Field(default=None, ge=0)
    seed: int | None = None
    num_ctx: int | None = Field(default=None, gt=0)
    keep_alive: str | int | None = None

    def options(self) -> dict[str, object] | None:
        values = self.model_dump(
            include={"temperature", "seed", "num_ctx"}, exclude_none=True
        )
        return values or None


class OllamaProvider:
    """Execute exactly one Ollama completion; tool execution is out of scope."""

    def __init__(
        self,
        client: OllamaClient,
        model: str,
        *,
        generation_settings: OllamaGenerationSettings | None = None,
        clock_ns: Callable[[], int] = time.perf_counter_ns,
    ) -> None:
        if not model.strip():
            raise ValueError("model must be a non-empty string")
        self._client = client
        self._model = model
        self._generation_settings = generation_settings
        self._clock_ns = clock_ns

    async def complete(self, request: ModelRequest) -> ModelResponse:
        """Return one non-streaming, provider-neutral completion."""

        started_ns = self._clock_ns()
        response = await self._client.chat(self._to_request(request))
        completed_ns = self._clock_ns()
        tool_calls = self._map_tool_calls(response.message.tool_calls)
        first_ns = (
            completed_ns
            if response.message.content.strip() or tool_calls
            else None
        )
        return ModelResponse(
            message=ChatMessage(
                role=ChatRole(response.message.role),
                content=response.message.content,
                tool_calls=tool_calls,
                tool_name=response.message.tool_name,
            ),
            usage=self._usage(response),
            performance=self._performance(started_ns, first_ns, completed_ns),
        )

    async def stream(
        self, request: ModelRequest
    ) -> AsyncIterator[ModelStreamChunk]:
        """Yield visible content/tool calls and attach metrics to the final chunk."""

        started_ns = self._clock_ns()
        first_ns: int | None = None
        async for response in self._client.stream_chat(self._to_request(request)):
            observed_ns = self._clock_ns()
            tool_calls = self._map_tool_calls(response.message.tool_calls)
            meaningful = bool(response.message.content.strip() or tool_calls)
            if meaningful and first_ns is None:
                first_ns = observed_ns

            if not meaningful and not response.done:
                # Thinking-only and empty chunks are intentionally not exposed.
                continue

            yield ModelStreamChunk(
                content=response.message.content,
                tool_calls=tool_calls,
                done=response.done,
                usage=self._usage(response) if response.done else None,
                performance=(
                    self._performance(started_ns, first_ns, observed_ns)
                    if response.done
                    else None
                ),
            )

    def _to_request(self, request: ModelRequest) -> OllamaChatRequest:
        return OllamaChatRequest(
            model=self._model,
            messages=[self._to_message(message) for message in request.messages],
            tools=to_ollama_tools(request.tools),
            stream=False,
            think=False,
            options=(
                None
                if self._generation_settings is None
                else self._generation_settings.options()
            ),
            keep_alive=(
                None
                if self._generation_settings is None
                else self._generation_settings.keep_alive
            ),
        )

    @staticmethod
    def _to_message(message: ChatMessage) -> OllamaChatMessage:
        return OllamaChatMessage(
            role=message.role.value,
            content=message.content,
            tool_name=message.tool_name,
            tool_calls=[
                OllamaToolCall(
                    function={"name": call.name, "arguments": call.arguments}
                )
                for call in message.tool_calls
            ],
        )

    @staticmethod
    def _map_tool_calls(calls: list[OllamaToolCall]) -> list[ModelToolCall]:
        return [
            ModelToolCall(
                name=call.function.name,
                arguments=call.function.arguments,
            )
            for call in calls
        ]

    @staticmethod
    def _usage(response: OllamaChatResponse) -> ModelUsageMetrics:
        return ModelUsageMetrics(
            total_duration_ns=response.total_duration,
            load_duration_ns=response.load_duration,
            prompt_eval_count=response.prompt_eval_count,
            prompt_eval_duration_ns=response.prompt_eval_duration,
            eval_count=response.eval_count,
            eval_duration_ns=response.eval_duration,
        )

    @staticmethod
    def _performance(
        started_ns: int,
        first_ns: int | None,
        completed_ns: int,
    ) -> ModelPerformanceMetrics:
        return ModelPerformanceMetrics(
            request_started_ns=started_ns,
            first_meaningful_chunk_ns=first_ns,
            completed_ns=completed_ns,
            client_wall_duration_ns=completed_ns - started_ns,
            time_to_first_chunk_ns=(
                None if first_ns is None else first_ns - started_ns
            ),
        )
