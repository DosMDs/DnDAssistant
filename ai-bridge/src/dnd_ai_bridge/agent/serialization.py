"""Deterministic model-facing serialization for 1C tool results."""

from __future__ import annotations

import json
from typing import Any

from ..models import ToolResult


def serialize_tool_result(result: ToolResult) -> str:
    """Return compact UTF-8-friendly JSON without natural-language rewriting."""

    payload = result.model_dump(mode="json")
    ordered_payload = {
        "success": payload["success"],
        "data": _sort_json_objects(payload["data"]),
        "error": _sort_json_objects(payload["error"]),
    }
    return json.dumps(
        ordered_payload,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _sort_json_objects(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _sort_json_objects(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_sort_json_objects(item) for item in value]
    return value
