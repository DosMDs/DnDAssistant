"""Versioned benchmark scenario and result contracts."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..models import ChatMessage, ToolDefinition


class BenchmarkModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ScenarioRole(StrEnum):
    TOOL_SELECTION = "tool_selection"
    TOOL_ARGUMENTS = "tool_arguments"
    CONTEXT_QA = "context_qa"
    CAMPAIGN_SUMMARY = "campaign_summary"


class BenchmarkMode(StrEnum):
    COLD = "cold"
    WARM = "warm"


class ScenarioExpectation(BenchmarkModel):
    expected_tool: str | None = None
    expected_arguments: dict[str, Any] | None = None
    arguments_subset: bool = False
    required_facts: list[str] = Field(default_factory=list)
    forbidden_facts: list[str] = Field(default_factory=list)


class Scenario(BenchmarkModel):
    schema_version: Literal["1"]
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]*$")
    version: int = Field(ge=1)
    role: ScenarioRole
    messages: list[ChatMessage] = Field(min_length=1)
    tools: list[ToolDefinition] = Field(default_factory=list)
    expectation: ScenarioExpectation

    @model_validator(mode="after")
    def validate_role_expectation(self) -> Scenario:
        if self.role in {ScenarioRole.TOOL_SELECTION, ScenarioRole.TOOL_ARGUMENTS}:
            if not self.expectation.expected_tool:
                raise ValueError("tool scenarios require expected_tool")
            if not self.tools:
                raise ValueError("tool scenarios require fixture tools")
        if self.role == ScenarioRole.TOOL_ARGUMENTS:
            if self.expectation.expected_arguments is None:
                raise ValueError("tool_arguments requires expected_arguments")
        if self.role in {ScenarioRole.CONTEXT_QA, ScenarioRole.CAMPAIGN_SUMMARY}:
            if not self.expectation.required_facts:
                raise ValueError("text scenarios require at least one required fact")
        return self


class ScenarioDocument(BenchmarkModel):
    schema_version: Literal["1"]
    tool_fixture_version: Literal["1"]
    scenarios: list[Scenario] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_scenario_ids(self) -> ScenarioDocument:
        keys = [(item.id, item.version) for item in self.scenarios]
        if len(keys) != len(set(keys)):
            raise ValueError("scenario id/version pairs must be unique")
        return self


class ScoringResult(BenchmarkModel):
    passed: bool
    checks: dict[str, bool] = Field(default_factory=dict)
    message: str | None = None


class RawMetrics(BenchmarkModel):
    client_wall_duration_ns: int | None = Field(default=None, ge=0)
    time_to_first_meaningful_chunk_ns: int | None = Field(default=None, ge=0)
    total_duration_ns: int | None = Field(default=None, ge=0)
    load_duration_ns: int | None = Field(default=None, ge=0)
    prompt_eval_count: int | None = Field(default=None, ge=0)
    prompt_eval_duration_ns: int | None = Field(default=None, ge=0)
    eval_count: int | None = Field(default=None, ge=0)
    eval_duration_ns: int | None = Field(default=None, ge=0)


class DerivedMetrics(BenchmarkModel):
    prompt_tokens_per_second: float | None = Field(default=None, ge=0)
    generation_tokens_per_second: float | None = Field(default=None, ge=0)


class EnvironmentMetadata(BenchmarkModel):
    os: str
    os_version: str
    architecture: str
    python_version: str
    ollama_version: str | None = None


class ModelMetadata(BenchmarkModel):
    requested_name: str
    reported_name: str | None = None
    digest: str | None = None
    parameter_size: str | None = None
    quantization: str | None = None
    capabilities: list[str] = Field(default_factory=list)
    context_metadata: dict[str, Any] = Field(default_factory=dict)
    allocated_context_length: int | None = Field(default=None, ge=0)
    size_vram: int | None = Field(default=None, ge=0)


class BenchmarkError(BenchmarkModel):
    code: str
    message: str


class BenchmarkResult(BenchmarkModel):
    schema_version: Literal["1"] = "1"
    run_id: str
    scenario_id: str
    scenario_version: int
    role: ScenarioRole
    model: str
    mode: BenchmarkMode
    repeat_index: int = Field(ge=1)
    generation_settings: dict[str, Any]
    environment: EnvironmentMetadata
    model_info: ModelMetadata
    raw_metrics: RawMetrics | None = None
    derived_metrics: DerivedMetrics | None = None
    scoring: ScoringResult | None = None
    error: BenchmarkError | None = None
