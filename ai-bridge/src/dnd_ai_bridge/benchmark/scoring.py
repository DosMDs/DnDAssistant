"""Local deterministic benchmark scoring; no model judge is involved."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from ..models import ModelToolCall
from .models import ScoringResult, Scenario, ScenarioRole


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(re.findall(r"[^\W_]+", normalized, flags=re.UNICODE))


def _contains(container: Any, expected: Any) -> bool:
    if isinstance(expected, dict):
        return isinstance(container, dict) and all(
            key in container and _contains(container[key], value)
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        return (
            isinstance(container, list)
            and len(container) == len(expected)
            and all(_contains(actual, wanted) for actual, wanted in zip(container, expected))
        )
    return container == expected


def score_scenario(
    scenario: Scenario,
    *,
    content: str,
    tool_calls: list[ModelToolCall],
) -> ScoringResult:
    expectation = scenario.expectation
    if scenario.role == ScenarioRole.TOOL_SELECTION:
        exact = (
            len(tool_calls) == 1
            and tool_calls[0].name == expectation.expected_tool
        )
        return ScoringResult(
            passed=exact,
            checks={"exact_expected_tool": exact},
            message=None if exact else "missing or unexpected tool call",
        )

    if scenario.role == ScenarioRole.TOOL_ARGUMENTS:
        tool_ok = (
            len(tool_calls) == 1
            and tool_calls[0].name == expectation.expected_tool
        )
        arguments_ok = False
        if tool_ok:
            actual = tool_calls[0].arguments
            expected = expectation.expected_arguments
            arguments_ok = (
                _contains(actual, expected)
                if expectation.arguments_subset
                else actual == expected
            )
        passed = tool_ok and arguments_ok
        return ScoringResult(
            passed=passed,
            checks={"exact_expected_tool": tool_ok, "arguments": arguments_ok},
            message=None if passed else "tool or arguments did not match",
        )

    normalized = normalize_text(content)
    required = {
        fact: normalize_text(fact) in normalized
        for fact in expectation.required_facts
    }
    forbidden = {
        fact: normalize_text(fact) not in normalized
        for fact in expectation.forbidden_facts
    }
    checks = {
        **{f"required:{fact}": value for fact, value in required.items()},
        **{f"forbidden:{fact}": value for fact, value in forbidden.items()},
    }
    passed = all(checks.values())
    return ScoringResult(
        passed=passed,
        checks=checks,
        message=None if passed else "required/forbidden fact check failed",
    )
