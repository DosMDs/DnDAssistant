"""Loading of versioned, synthetic benchmark scenarios and tool fixtures."""

from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

from ..models import ToolDefinition
from .models import Scenario, ScenarioDocument, ScenarioRole


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def load_scenario_document(path: str | Path) -> ScenarioDocument:
    return ScenarioDocument.model_validate(_read_json(Path(path)))


def load_tool_fixtures(path: str | Path) -> list[ToolDefinition]:
    data = _read_json(Path(path))
    if not isinstance(data, dict) or data.get("schema_version") != "1":
        raise ValueError("unsupported tool fixture schema_version")
    return TypeAdapter(list[ToolDefinition]).validate_python(data.get("tools"))


def load_builtin_scenarios(
    role: ScenarioRole | None = None,
) -> list[Scenario]:
    data_dir = files("dnd_ai_bridge.benchmark.data")
    tool_data = json.loads(data_dir.joinpath("tools.v1.json").read_text("utf-8"))
    if tool_data.get("schema_version") != "1":
        raise ValueError("unsupported built-in tool fixture version")
    tools = TypeAdapter(list[ToolDefinition]).validate_python(tool_data["tools"])
    raw = json.loads(data_dir.joinpath("scenarios.v1.json").read_text("utf-8"))
    for item in raw.get("scenarios", []):
        if item.get("role") in {"tool_selection", "tool_arguments"}:
            item["tools"] = [tool.model_dump(mode="json") for tool in tools]
    document = ScenarioDocument.model_validate(raw)
    if role is None:
        return document.scenarios
    return [scenario for scenario in document.scenarios if scenario.role == role]
