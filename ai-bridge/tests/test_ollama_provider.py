from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from typing import Any

import httpx
import pytest

from dnd_ai_bridge.config import OllamaSettings
from dnd_ai_bridge.models import (
    ChatMessage,
    ChatRole,
    ModelRequest,
    ToolDefinition,
)
from dnd_ai_bridge.ollama_client import OllamaClient
from dnd_ai_bridge.ollama_provider import OllamaProvider


def clock(values: list[int]) -> Callable[[], int]:
    iterator: Iterator[int] = iter(values)
    return lambda: next(iterator)


def settings() -> OllamaSettings:
    return OllamaSettings(ollama_timeout_seconds=1)


def response_body(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": "qwen3:8b",
        "created_at": "2026-08-19T12:00:00Z",
        "message": {"role": "assistant", "content": "Ответ"},
        "done": True,
        "total_duration": 1_000,
        "load_duration": 100,
        "prompt_eval_count": 10,
        "prompt_eval_duration": 200,
        "eval_count": 5,
        "eval_duration": 300,
    }
    body.update(overrides)
    return body


@pytest.mark.asyncio
async def test_provider_maps_neutral_request_response_tools_and_metrics() -> None:
    received: dict[str, Any] = {}

    def handler(http_request: httpx.Request) -> httpx.Response:
        nonlocal received
        received = json.loads(http_request.content.decode("utf-8"))
        return httpx.Response(200, json=response_body(), request=http_request)

    tool = ToolDefinition(
        name="search_entities",
        description="Поиск",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    )
    neutral_request = ModelRequest(
        messages=[ChatMessage(role=ChatRole.USER, content="Найди Торвальда")],
        tools=[tool],
    )
    async with OllamaClient(
        settings(), transport=httpx.MockTransport(handler)
    ) as client:
        provider = OllamaProvider(client, "qwen3:8b", clock_ns=clock([100, 550]))
        response = await provider.complete(neutral_request)

    assert received["messages"] == [
        {"role": "user", "content": "Найди Торвальда"}
    ]
    assert received["tools"][0]["function"]["name"] == "search_entities"
    assert response.message.role == ChatRole.ASSISTANT
    assert response.message.content == "Ответ"
    assert response.usage.total_duration_ns == 1_000
    assert response.usage.prompt_eval_count == 10
    assert response.usage.eval_duration_ns == 300
    assert response.performance is not None
    assert response.performance.client_wall_duration_ns == 450
    assert response.performance.time_to_first_chunk_ns == 450


@pytest.mark.asyncio
async def test_provider_maps_tool_call_without_executing_it_or_exposing_thinking() -> None:
    body = response_body(
        message={
            "role": "assistant",
            "content": "",
            "thinking": "внутреннее рассуждение",
            "tool_calls": [
                {
                    "function": {
                        "name": "get_entity",
                        "arguments": {"id": "42", "type": "npc"},
                    }
                }
            ],
        }
    )
    async with OllamaClient(
        settings(),
        transport=httpx.MockTransport(
            lambda req: httpx.Response(200, json=body, request=req)
        ),
    ) as client:
        provider = OllamaProvider(client, "qwen3:8b", clock_ns=clock([10, 20]))
        response = await provider.complete(
            ModelRequest(messages=[ChatMessage(role=ChatRole.USER, content="Найди")])
        )

    assert response.message.content == ""
    assert response.message.tool_calls[0].name == "get_entity"
    assert response.message.tool_calls[0].arguments == {"id": "42", "type": "npc"}
    assert "thinking" not in response.model_dump(mode="json")


@pytest.mark.asyncio
async def test_provider_stream_measures_first_meaningful_visible_chunk() -> None:
    chunks = [
        response_body(
            message={
                "role": "assistant",
                "content": "",
                "thinking": "скрыто",
            },
            done=False,
        ),
        response_body(
            message={"role": "assistant", "content": "Привет"},
            done=False,
        ),
        response_body(message={"role": "assistant", "content": ""}),
    ]
    ndjson = "\n".join(json.dumps(item, ensure_ascii=False) for item in chunks)
    async with OllamaClient(
        settings(),
        transport=httpx.MockTransport(
            lambda req: httpx.Response(
                200, content=ndjson.encode("utf-8"), request=req
            )
        ),
    ) as client:
        provider = OllamaProvider(
            client,
            "qwen3:8b",
            clock_ns=clock([100, 200, 350, 500]),
        )
        visible = [
            chunk
            async for chunk in provider.stream(
                ModelRequest(
                    messages=[ChatMessage(role=ChatRole.USER, content="Привет")]
                )
            )
        ]

    assert [chunk.content for chunk in visible] == ["Привет", ""]
    assert visible[-1].done is True
    assert visible[-1].usage is not None
    assert visible[-1].usage.eval_count == 5
    assert visible[-1].performance is not None
    assert visible[-1].performance.request_started_ns == 100
    assert visible[-1].performance.first_meaningful_chunk_ns == 350
    assert visible[-1].performance.completed_ns == 500
    assert visible[-1].performance.client_wall_duration_ns == 400
    assert visible[-1].performance.time_to_first_chunk_ns == 250
