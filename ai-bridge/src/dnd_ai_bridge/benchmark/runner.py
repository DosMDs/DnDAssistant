"""Cold/warm orchestration for streaming, repeatable Ollama benchmarks."""

from __future__ import annotations

import re
import uuid
from collections.abc import Callable, Sequence
from typing import Any

from ..models import ModelRequest, ModelStreamChunk, ModelToolCall
from ..ollama_client import OllamaClient
from ..ollama_models import OllamaRunningModel, OllamaShowModelResponse
from ..ollama_provider import OllamaGenerationSettings, OllamaProvider
from .environment import local_environment, model_metadata
from .models import (
    BenchmarkError,
    BenchmarkMode,
    BenchmarkResult,
    DerivedMetrics,
    EnvironmentMetadata,
    RawMetrics,
    Scenario,
)
from .output import JsonlResultWriter
from .scoring import score_scenario

ProviderFactory = Callable[..., OllamaProvider]


class InvalidBenchmarkState(RuntimeError):
    pass


def reject_cloud_model(model: str) -> str:
    """Reject identifiers that explicitly select a cloud/remote model."""

    candidate = model.strip()
    if not candidate:
        raise ValueError("model must be a non-empty string")
    lowered = candidate.casefold()
    explicit_cloud = (
        "://" in lowered
        or re.search(r"(?:^|[/:@_.-])cloud(?:$|[/:@_.-])", lowered) is not None
        or lowered.startswith("cloud.")
    )
    if explicit_cloud:
        raise ValueError("cloud model identifiers are forbidden in offline mode")
    return candidate


def calculate_derived_metrics(raw: RawMetrics) -> DerivedMetrics:
    def rate(count: int | None, duration_ns: int | None) -> float | None:
        if count is None or duration_ns is None or duration_ns == 0:
            return None
        return count * 1_000_000_000 / duration_ns

    return DerivedMetrics(
        prompt_tokens_per_second=rate(
            raw.prompt_eval_count, raw.prompt_eval_duration_ns
        ),
        generation_tokens_per_second=rate(raw.eval_count, raw.eval_duration_ns),
    )


def _same_model(requested: str, actual: str) -> bool:
    def canonical(value: str) -> str:
        normalized = value.casefold()
        return normalized[:-7] if normalized.endswith(":latest") else normalized

    return canonical(requested) == canonical(actual)


def _find_running(
    model: str, running: Sequence[OllamaRunningModel]
) -> OllamaRunningModel | None:
    return next(
        (
            item
            for item in running
            if any(
                value and _same_model(model, value)
                for value in (item.name, item.model)
            )
        ),
        None,
    )


def _raw_metrics(final: ModelStreamChunk) -> RawMetrics:
    usage = final.usage
    performance = final.performance
    return RawMetrics(
        client_wall_duration_ns=(
            None if performance is None else performance.client_wall_duration_ns
        ),
        time_to_first_meaningful_chunk_ns=(
            None if performance is None else performance.time_to_first_chunk_ns
        ),
        total_duration_ns=None if usage is None else usage.total_duration_ns,
        load_duration_ns=None if usage is None else usage.load_duration_ns,
        prompt_eval_count=None if usage is None else usage.prompt_eval_count,
        prompt_eval_duration_ns=(
            None if usage is None else usage.prompt_eval_duration_ns
        ),
        eval_count=None if usage is None else usage.eval_count,
        eval_duration_ns=None if usage is None else usage.eval_duration_ns,
    )


class BenchmarkRunner:
    """Run scenarios while proving cold/warm state through ``/api/ps``."""

    def __init__(
        self,
        client: OllamaClient,
        writer: JsonlResultWriter,
        *,
        provider_factory: ProviderFactory = OllamaProvider,
        run_id_factory: Callable[[], str] = lambda: str(uuid.uuid4()),
    ) -> None:
        self._client = client
        self._writer = writer
        self._provider_factory = provider_factory
        self._run_id_factory = run_id_factory

    async def run(
        self,
        *,
        model: str,
        scenarios: Sequence[Scenario],
        repeat: int,
        cold: bool,
        warm: bool,
        generation_settings: OllamaGenerationSettings | None = None,
    ) -> list[BenchmarkResult]:
        model = reject_cloud_model(model)
        if repeat < 1:
            raise ValueError("repeat must be at least 1")
        if not cold and not warm:
            raise ValueError("at least one of cold or warm must be enabled")
        settings = generation_settings or OllamaGenerationSettings(
            temperature=0.0, seed=0
        )

        version = await self._client.version()
        shown = await self._client.show_model(model, verbose=True)
        environment = local_environment(version.version)
        results: list[BenchmarkResult] = []
        run_id = self._run_id_factory()
        modes = [
            mode
            for mode, enabled in (
                (BenchmarkMode.COLD, cold),
                (BenchmarkMode.WARM, warm),
            )
            if enabled
        ]
        for scenario in scenarios:
            for mode in modes:
                for repeat_index in range(1, repeat + 1):
                    result = await self._run_one(
                        model=model,
                        run_id=run_id,
                        scenario=scenario,
                        mode=mode,
                        repeat_index=repeat_index,
                        settings=settings,
                        environment=environment,
                        shown=shown,
                    )
                    self._writer.append(result)
                    results.append(result)
        return results

    async def _run_one(
        self,
        *,
        model: str,
        run_id: str,
        scenario: Scenario,
        mode: BenchmarkMode,
        repeat_index: int,
        settings: OllamaGenerationSettings,
        environment: EnvironmentMetadata,
        shown: OllamaShowModelResponse,
    ) -> BenchmarkResult:
        base: dict[str, Any] = {
            "run_id": run_id,
            "scenario_id": scenario.id,
            "scenario_version": scenario.version,
            "role": scenario.role,
            "model": model,
            "mode": mode,
            "repeat_index": repeat_index,
            "generation_settings": settings.model_dump(
                mode="json", exclude_none=True
            ),
            "environment": environment,
        }
        try:
            provider = self._provider_factory(
                self._client, model, generation_settings=settings
            )
            request = ModelRequest(messages=scenario.messages, tools=scenario.tools)
            if mode == BenchmarkMode.COLD:
                unloaded = await self._client.unload_model(model)
                if not unloaded.done:
                    raise InvalidBenchmarkState("Ollama did not confirm unload")
                await self._verify_loaded(model, expected=False)
            else:
                await self._consume_stream(provider, request)
                await self._verify_loaded(model, expected=True)

            chunks = await self._consume_stream(provider, request)
            final = next((chunk for chunk in reversed(chunks) if chunk.done), None)
            if final is None:
                raise InvalidBenchmarkState("measured stream has no terminal chunk")
            content = "".join(chunk.content for chunk in chunks)
            calls: list[ModelToolCall] = []
            for chunk in chunks:
                calls.extend(chunk.tool_calls)
            raw = _raw_metrics(final)
            running = await self._client.running_models()
            loaded = _find_running(model, running.models)
            info = model_metadata(model, shown, loaded)
            return BenchmarkResult(
                **base,
                model_info=info,
                raw_metrics=raw,
                derived_metrics=calculate_derived_metrics(raw),
                scoring=score_scenario(
                    scenario, content=content, tool_calls=calls
                ),
            )
        except Exception as exc:
            code = (
                "invalid_state"
                if isinstance(exc, InvalidBenchmarkState)
                else "run_failed"
            )
            return BenchmarkResult(
                **base,
                model_info=model_metadata(model, shown, None),
                error=BenchmarkError(code=code, message=str(exc)),
            )

    async def _verify_loaded(self, model: str, *, expected: bool) -> None:
        try:
            response = await self._client.running_models()
        except Exception as exc:
            raise InvalidBenchmarkState(
                "unable to confirm model state through /api/ps"
            ) from exc
        actual = _find_running(model, response.models) is not None
        if actual != expected:
            wanted = "loaded" if expected else "unloaded"
            raise InvalidBenchmarkState(f"model is not confirmed {wanted}")

    @staticmethod
    async def _consume_stream(
        provider: OllamaProvider, request: ModelRequest
    ) -> list[ModelStreamChunk]:
        return [chunk async for chunk in provider.stream(request)]
