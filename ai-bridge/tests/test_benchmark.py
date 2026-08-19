from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from dnd_ai_bridge.benchmark.environment import local_environment, model_metadata
from dnd_ai_bridge.benchmark.models import (
    BenchmarkMode,
    EnvironmentMetadata,
    RawMetrics,
    Scenario,
    ScenarioRole,
)
from dnd_ai_bridge.benchmark.output import JsonlResultWriter
from dnd_ai_bridge.benchmark.runner import (
    BenchmarkRunner,
    calculate_derived_metrics,
    reject_cloud_model,
)
from dnd_ai_bridge.benchmark.scenarios import load_builtin_scenarios
from dnd_ai_bridge.benchmark.scoring import normalize_text, score_scenario
from dnd_ai_bridge.models import (
    ChatMessage,
    ChatRole,
    ModelPerformanceMetrics,
    ModelStreamChunk,
    ModelToolCall,
    ModelUsageMetrics,
    ToolDefinition,
)
from dnd_ai_bridge.ollama_models import (
    OllamaModelDetails,
    OllamaRunningModel,
    OllamaRunningModelsResponse,
    OllamaShowModelResponse,
    OllamaUnloadResponse,
    OllamaVersionResponse,
)


def scenario(
    role: ScenarioRole,
    expectation: dict[str, Any],
    *,
    tools: bool = False,
) -> Scenario:
    fixture_tools = (
        [
            ToolDefinition(
                name="lookup",
                description="lookup",
                parameters={"type": "object"},
            )
        ]
        if tools
        else []
    )
    return Scenario.model_validate(
        {
            "schema_version": "1",
            "id": f"test.{role.value}",
            "version": 1,
            "role": role,
            "messages": [{"role": "user", "content": "test"}],
            "tools": [item.model_dump(mode="json") for item in fixture_tools],
            "expectation": expectation,
        }
    )


def test_builtin_scenarios_cover_all_roles_and_are_versioned() -> None:
    scenarios = load_builtin_scenarios()
    assert {item.role for item in scenarios} == set(ScenarioRole)
    assert all(item.schema_version == "1" and item.version == 1 for item in scenarios)
    assert all(item.tools for item in scenarios if "tool." in item.id)


def test_scenario_rejects_unknown_schema_version() -> None:
    data = {
        "schema_version": "2",
        "id": "bad.version",
        "version": 1,
        "role": "context_qa",
        "messages": [{"role": "user", "content": "x"}],
        "expectation": {"required_facts": ["x"]},
    }
    with pytest.raises(ValidationError):
        Scenario.model_validate(data)


def test_tool_selection_requires_one_exact_tool() -> None:
    item = scenario(
        ScenarioRole.TOOL_SELECTION, {"expected_tool": "lookup"}, tools=True
    )
    assert score_scenario(
        item,
        content="",
        tool_calls=[ModelToolCall(name="lookup", arguments={})],
    ).passed
    assert not score_scenario(item, content="", tool_calls=[]).passed
    assert not score_scenario(
        item,
        content="",
        tool_calls=[
            ModelToolCall(name="lookup", arguments={}),
            ModelToolCall(name="invented", arguments={}),
        ],
    ).passed


@pytest.mark.parametrize("subset", [False, True])
def test_tool_arguments_structural_scoring(subset: bool) -> None:
    item = scenario(
        ScenarioRole.TOOL_ARGUMENTS,
        {
            "expected_tool": "lookup",
            "expected_arguments": {"query": "spire", "nested": {"limit": 2}},
            "arguments_subset": subset,
        },
        tools=True,
    )
    arguments = {"query": "spire", "nested": {"limit": 2}, "extra": True}
    result = score_scenario(
        item,
        content="",
        tool_calls=[ModelToolCall(name="lookup", arguments=arguments)],
    )
    assert result.passed is subset


@pytest.mark.parametrize(
    "role", [ScenarioRole.CONTEXT_QA, ScenarioRole.CAMPAIGN_SUMMARY]
)
def test_text_scoring_normalizes_and_checks_required_and_forbidden_facts(
    role: ScenarioRole,
) -> None:
    item = scenario(
        role,
        {"required_facts": ["Glass Harbor"], "forbidden_facts": ["Moonport"]},
    )
    assert normalize_text("GLASS—Harbor") == "glass harbor"
    assert score_scenario(
        item, content="They met at glass, harbor.", tool_calls=[]
    ).passed
    assert not score_scenario(
        item, content="They met at Glass Harbor, then Moonport.", tool_calls=[]
    ).passed


def test_metric_rates_handle_values_zero_and_none() -> None:
    derived = calculate_derived_metrics(
        RawMetrics(
            prompt_eval_count=20,
            prompt_eval_duration_ns=2_000_000_000,
            eval_count=4,
            eval_duration_ns=0,
        )
    )
    assert derived.prompt_tokens_per_second == 10.0
    assert derived.generation_tokens_per_second is None
    assert calculate_derived_metrics(RawMetrics()).prompt_tokens_per_second is None


def test_environment_and_model_metadata_are_portable_and_typed() -> None:
    environment = local_environment("0.12.6")
    assert environment.os
    assert environment.python_version
    shown = OllamaShowModelResponse(
        details=OllamaModelDetails(
            parameter_size="8B", quantization_level="Q4_K_M"
        ),
        capabilities=["completion", "tools"],
        model_info={"qwen.context_length": 32768, "qwen.embedding_length": 4096},
    )
    running = OllamaRunningModel(
        name="qwen3:8b", digest="abc", context_length=8192, size_vram=42
    )
    info = model_metadata("qwen3:8b", shown, running)
    assert info.parameter_size == "8B"
    assert info.context_metadata == {"qwen.context_length": 32768}
    assert info.allocated_context_length == 8192
    assert info.size_vram == 42


@pytest.mark.parametrize(
    "model", ["gpt-oss:cloud", "gpt-oss:120b-cloud", "https://cloud/model"]
)
def test_cloud_model_identifiers_are_rejected(model: str) -> None:
    with pytest.raises(ValueError, match="cloud"):
        reject_cloud_model(model)
    assert reject_cloud_model("qwen3:8b") == "qwen3:8b"


class FakeClient:
    def __init__(
        self, *, fail_ps: bool = False, prevent_load: bool = False
    ) -> None:
        self.loaded = False
        self.fail_ps = fail_ps
        self.prevent_load = prevent_load
        self.ps_calls = 0

    async def version(self) -> OllamaVersionResponse:
        return OllamaVersionResponse(version="test")

    async def show_model(self, model: str, *, verbose: bool = False) -> OllamaShowModelResponse:
        assert verbose
        return OllamaShowModelResponse(
            details=OllamaModelDetails(
                parameter_size="1B", quantization_level="Q4"
            ),
            capabilities=["completion"],
            model_info={"test.context_length": 4096},
        )

    async def unload_model(self, model: str) -> OllamaUnloadResponse:
        self.loaded = False
        return OllamaUnloadResponse(model=model, done=True)

    async def running_models(self) -> OllamaRunningModelsResponse:
        self.ps_calls += 1
        if self.fail_ps:
            raise RuntimeError("ps unavailable")
        models = (
            [
                OllamaRunningModel(
                    name="local:1b", context_length=2048, size_vram=100
                )
            ]
            if self.loaded
            else []
        )
        return OllamaRunningModelsResponse(models=models)


class FakeProvider:
    def __init__(self, client: FakeClient, counter: list[int], fail_at: int | None) -> None:
        self.client = client
        self.counter = counter
        self.fail_at = fail_at

    async def stream(self, request: object):
        self.counter[0] += 1
        call = self.counter[0]
        self.client.loaded = not self.client.prevent_load
        if call == self.fail_at:
            raise RuntimeError("synthetic failure")
        yield ModelStreamChunk(
            content="Glass Harbor",
            done=True,
            usage=ModelUsageMetrics(
                total_duration_ns=100,
                load_duration_ns=10,
                prompt_eval_count=20,
                prompt_eval_duration_ns=2_000_000_000,
                eval_count=5,
                eval_duration_ns=1_000_000_000,
            ),
            performance=ModelPerformanceMetrics(
                request_started_ns=0,
                first_meaningful_chunk_ns=call,
                completed_ns=call,
                client_wall_duration_ns=call,
                time_to_first_chunk_ns=call,
            ),
        )


def provider_factory(counter: list[int], fail_at: int | None = None):
    def factory(client: FakeClient, model: str, **kwargs: object) -> FakeProvider:
        return FakeProvider(client, counter, fail_at)

    return factory


@pytest.mark.asyncio
async def test_runner_verifies_cold_warm_repeats_and_excludes_warmups(
    tmp_path: Path,
) -> None:
    path = tmp_path / "results.jsonl"
    client = FakeClient()
    counter = [0]
    runner = BenchmarkRunner(
        client,  # type: ignore[arg-type]
        JsonlResultWriter(path),
        provider_factory=provider_factory(counter),  # type: ignore[arg-type]
        run_id_factory=lambda: f"run-{counter[0]}",
    )
    results = await runner.run(
        model="local:1b",
        scenarios=[
            scenario(
                ScenarioRole.CONTEXT_QA,
                {"required_facts": ["Glass Harbor"]},
            )
        ],
        repeat=2,
        cold=True,
        warm=True,
    )

    assert len(results) == 4
    assert [item.mode for item in results] == [
        BenchmarkMode.COLD,
        BenchmarkMode.COLD,
        BenchmarkMode.WARM,
        BenchmarkMode.WARM,
    ]
    assert [item.raw_metrics.client_wall_duration_ns for item in results if item.raw_metrics] == [1, 2, 4, 6]
    assert counter[0] == 6  # two warm-up streams are not result records
    lines = path.read_text("utf-8").splitlines()
    assert len(lines) == 4
    assert all(json.loads(line)["schema_version"] == "1" for line in lines)
    assert results[0].model_info.allocated_context_length == 2048


@pytest.mark.asyncio
async def test_unconfirmed_state_is_invalid_and_written_to_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "invalid.jsonl"
    runner = BenchmarkRunner(
        FakeClient(fail_ps=True),  # type: ignore[arg-type]
        JsonlResultWriter(path),
        provider_factory=provider_factory([0]),  # type: ignore[arg-type]
    )
    results = await runner.run(
        model="local:1b",
        scenarios=[
            scenario(ScenarioRole.CONTEXT_QA, {"required_facts": ["x"]})
        ],
        repeat=1,
        cold=True,
        warm=False,
    )
    assert results[0].error is not None
    assert results[0].error.code == "invalid_state"
    assert json.loads(path.read_text("utf-8"))["error"]["code"] == "invalid_state"


@pytest.mark.asyncio
async def test_warm_run_is_invalid_when_warmup_did_not_load_model(
    tmp_path: Path,
) -> None:
    runner = BenchmarkRunner(
        FakeClient(prevent_load=True),  # type: ignore[arg-type]
        JsonlResultWriter(tmp_path / "warm-invalid.jsonl"),
        provider_factory=provider_factory([0]),  # type: ignore[arg-type]
    )
    results = await runner.run(
        model="local:1b",
        scenarios=[
            scenario(ScenarioRole.CONTEXT_QA, {"required_facts": ["x"]})
        ],
        repeat=1,
        cold=False,
        warm=True,
    )
    assert results[0].error is not None
    assert results[0].error.code == "invalid_state"
    assert results[0].raw_metrics is None


@pytest.mark.asyncio
async def test_partial_failure_preserves_previous_jsonl_record(tmp_path: Path) -> None:
    path = tmp_path / "partial.jsonl"
    counter = [0]
    runner = BenchmarkRunner(
        FakeClient(),  # type: ignore[arg-type]
        JsonlResultWriter(path),
        provider_factory=provider_factory(counter, fail_at=2),  # type: ignore[arg-type]
    )
    results = await runner.run(
        model="local:1b",
        scenarios=[
            scenario(
                ScenarioRole.CONTEXT_QA,
                {"required_facts": ["Glass Harbor"]},
            )
        ],
        repeat=2,
        cold=True,
        warm=False,
    )
    lines = [json.loads(line) for line in path.read_text("utf-8").splitlines()]
    assert results[0].error is None
    assert results[1].error is not None
    assert len(lines) == 2
    assert lines[0]["scoring"]["passed"] is True
    assert lines[1]["error"]["code"] == "run_failed"
