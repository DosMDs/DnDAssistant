from __future__ import annotations

import os
from typing import Any

import pytest

from dnd_ai_bridge.config import BridgeSettings
from dnd_ai_bridge.models import ToolResult
from dnd_ai_bridge.onec_client import OneCClient

pytestmark = pytest.mark.integration

EXPECTED_TOOLS = {
    "get_current_context",
    "search_entities",
    "get_entity",
    "get_relations",
    "get_calendar_agenda",
}


def _live_settings() -> BridgeSettings:
    required = (
        "DND_ONEC_BASE_URL",
        "DND_ONEC_USERNAME",
        "DND_ONEC_PASSWORD",
    )
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        pytest.skip("live 1C credentials are not configured")
    return BridgeSettings(_env_file=None)


@pytest.mark.asyncio
async def test_live_health() -> None:
    async with OneCClient(_live_settings()) as client:
        health = await client.health()

    assert health.status == "ok"
    assert health.api_version == "1"


@pytest.mark.asyncio
async def test_live_tools_current_contract() -> None:
    async with OneCClient(_live_settings()) as client:
        tools = await client.list_tools()

    by_name = {tool.name: tool for tool in tools}
    assert EXPECTED_TOOLS <= by_name.keys()
    assert all(by_name[name].read_only for name in EXPECTED_TOOLS)


def _first_entity_reference(result: ToolResult) -> tuple[Any, Any] | None:
    """Extract the dynamic entity identity returned by search_entities."""

    if not result.success or not isinstance(result.data, dict):
        return None
    for collection_name in ("items", "entities", "results", "candidates"):
        collection = result.data.get(collection_name)
        if not isinstance(collection, list):
            continue
        for candidate in collection:
            if not isinstance(candidate, dict):
                continue
            entity_id = candidate.get("id")
            entity_type = candidate.get("type")
            if entity_id is not None and entity_type is not None:
                return entity_id, entity_type
    return None


@pytest.mark.asyncio
async def test_live_read_only_tool_flow() -> None:
    async with OneCClient(_live_settings()) as client:
        context = await client.call_tool("get_current_context", {})
        assert context.success, context.error

        search = await client.call_tool("search_entities", {"query": "а", "limit": 1})
        assert search.success, search.error

        entity_reference = _first_entity_reference(search)
        if entity_reference is not None:
            entity_id, entity_type = entity_reference
            arguments = {"id": entity_id, "type": entity_type}
            entity = await client.call_tool("get_entity", arguments)
            relations = await client.call_tool(
                "get_relations", arguments
            )
            assert entity.success, entity.error
            assert relations.success, relations.error

        agenda = await client.call_tool("get_calendar_agenda", {})
        assert agenda.success, agenda.error
