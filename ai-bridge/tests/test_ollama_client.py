from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from dnd_ai_bridge.config import OllamaSettings
from dnd_ai_bridge.errors import (
    OllamaConnectionError,
    OllamaHTTPStatusError,
    OllamaProtocolError,
    OllamaTimeoutError,
)
from dnd_ai_bridge.ollama_client import OllamaClient
from dnd_ai_bridge.ollama_models import OllamaChatMessage, OllamaChatRequest


def settings() -> OllamaSettings:
    return OllamaSettings(
        ollama_base_url="http://127.0.0.1:11434/",
        ollama_timeout_seconds=1,
    )


def request(content: str = "Привет") -> OllamaChatRequest:
    return OllamaChatRequest(
        model="qwen3:8b",
        messages=[OllamaChatMessage(role="user", content=content)],
    )


def chat_body(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": "qwen3:8b",
        "created_at": "2026-08-19T12:00:00Z",
        "message": {"role": "assistant", "content": "Здравствуйте!"},
        "done": True,
        "done_reason": "stop",
        "total_duration": 100,
        "load_duration": 20,
        "prompt_eval_count": 7,
        "prompt_eval_duration": 30,
        "eval_count": 4,
        "eval_duration": 40,
    }
    body.update(overrides)
    return body


@pytest.mark.asyncio
async def test_chat_posts_non_streaming_request_and_parses_response() -> None:
    received: dict[str, Any] = {}

    def handler(http_request: httpx.Request) -> httpx.Response:
        nonlocal received
        assert http_request.method == "POST"
        assert http_request.url.path == "/api/chat"
        assert http_request.headers.get("Authorization") is None
        received = json.loads(http_request.content.decode("utf-8"))
        return httpx.Response(200, json=chat_body(), request=http_request)

    async with OllamaClient(
        settings(), transport=httpx.MockTransport(handler)
    ) as client:
        response = await client.chat(request())

    assert received["stream"] is False
    assert received["think"] is False
    assert received["model"] == "qwen3:8b"
    assert response.message.content == "Здравствуйте!"
    assert response.done is True


@pytest.mark.asyncio
async def test_stream_chat_parses_ndjson_chunks() -> None:
    chunks = [
        chat_body(
            message={"role": "assistant", "content": "При"},
            done=False,
            total_duration=None,
        ),
        chat_body(message={"role": "assistant", "content": "вет"}),
    ]
    ndjson = "\n".join(json.dumps(item, ensure_ascii=False) for item in chunks) + "\n"

    def handler(http_request: httpx.Request) -> httpx.Response:
        payload = json.loads(http_request.content.decode("utf-8"))
        assert payload["stream"] is True
        return httpx.Response(
            200,
            content=ndjson.encode("utf-8"),
            headers={"Content-Type": "application/x-ndjson"},
            request=http_request,
        )

    async with OllamaClient(
        settings(), transport=httpx.MockTransport(handler)
    ) as client:
        responses = [chunk async for chunk in client.stream_chat(request())]

    assert [chunk.message.content for chunk in responses] == ["При", "вет"]
    assert responses[-1].done is True
    assert responses[-1].eval_count == 4


@pytest.mark.asyncio
async def test_tool_call_response_is_typed() -> None:
    body = chat_body(
        message={
            "role": "assistant",
            "content": "",
            "thinking": "не показывать",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "search_entities",
                        "arguments": {"query": "Торвальд"},
                    },
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
        response = await client.chat(request())

    call = response.message.tool_calls[0]
    assert call.function.name == "search_entities"
    assert call.function.arguments == {"query": "Торвальд"}
    assert response.message.thinking == "не показывать"


@pytest.mark.asyncio
async def test_chat_rejects_malformed_json() -> None:
    async with OllamaClient(
        settings(),
        transport=httpx.MockTransport(
            lambda req: httpx.Response(200, content=b"{", request=req)
        ),
    ) as client:
        with pytest.raises(OllamaProtocolError, match="not valid JSON"):
            await client.chat(request())


@pytest.mark.asyncio
async def test_stream_rejects_malformed_ndjson() -> None:
    async with OllamaClient(
        settings(),
        transport=httpx.MockTransport(
            lambda req: httpx.Response(
                200, content=b'{"broken":}\n', request=req
            )
        ),
    ) as client:
        with pytest.raises(OllamaProtocolError, match="invalid NDJSON"):
            _ = [chunk async for chunk in client.stream_chat(request())]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("raised", "expected"),
    [
        (httpx.ReadTimeout("slow"), OllamaTimeoutError),
        (httpx.ConnectError("offline"), OllamaConnectionError),
    ],
)
async def test_expected_network_errors_are_structured(
    raised: httpx.RequestError,
    expected: type[Exception],
) -> None:
    def handler(http_request: httpx.Request) -> httpx.Response:
        raised.request = http_request
        raise raised

    async with OllamaClient(
        settings(), transport=httpx.MockTransport(handler)
    ) as client:
        with pytest.raises(expected):
            await client.chat(request())


@pytest.mark.asyncio
async def test_unexpected_http_status_is_structured() -> None:
    async with OllamaClient(
        settings(),
        transport=httpx.MockTransport(
            lambda req: httpx.Response(
                404, json={"error": "model not found"}, request=req
            )
        ),
    ) as client:
        with pytest.raises(OllamaHTTPStatusError) as error:
            await client.chat(request())

    assert error.value.code == "unexpected_http_status"
    assert error.value.http_status == 404
    assert str(error.value) == "model not found"


@pytest.mark.asyncio
async def test_utf8_is_preserved_in_request_and_response() -> None:
    phrase = "Торвальд Железнорукий — где он?"

    def handler(http_request: httpx.Request) -> httpx.Response:
        assert phrase.encode("utf-8") in http_request.content
        body = chat_body(
            message={"role": "assistant", "content": "Он в Глубоководье."}
        )
        return httpx.Response(200, json=body, request=http_request)

    async with OllamaClient(
        settings(), transport=httpx.MockTransport(handler)
    ) as client:
        response = await client.chat(request(phrase))

    assert response.message.content == "Он в Глубоководье."


@pytest.mark.asyncio
async def test_diagnostic_endpoints_return_typed_model_information() -> None:
    def handler(http_request: httpx.Request) -> httpx.Response:
        if http_request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.12.6"}, request=http_request)
        if http_request.url.path == "/api/tags":
            return httpx.Response(
                200,
                json={"models": [{"name": "qwen3:8b", "size": 42}]},
                request=http_request,
            )
        assert http_request.url.path == "/api/show"
        assert json.loads(http_request.content) == {
            "model": "qwen3:8b",
            "verbose": False,
        }
        return httpx.Response(
            200,
            json={
                "capabilities": ["completion", "tools"],
                "details": {"parameter_size": "8.2B"},
                "model_info": {"general.architecture": "qwen3"},
            },
            request=http_request,
        )

    async with OllamaClient(
        settings(), transport=httpx.MockTransport(handler)
    ) as client:
        version = await client.version()
        models = await client.list_models()
        shown = await client.show_model("qwen3:8b")

    assert version.version == "0.12.6"
    assert models.models[0].name == "qwen3:8b"
    assert shown.details is not None
    assert shown.details.parameter_size == "8.2B"
    assert shown.model_info["general.architecture"] == "qwen3"


@pytest.mark.asyncio
async def test_running_models_endpoint_is_typed() -> None:
    def handler(http_request: httpx.Request) -> httpx.Response:
        assert http_request.method == "GET"
        assert http_request.url.path == "/api/ps"
        return httpx.Response(
            200,
            json={
                "models": [
                    {
                        "name": "qwen3:8b",
                        "model": "qwen3:8b",
                        "digest": "sha256:abc",
                        "size": 5_000,
                        "size_vram": 4_000,
                        "context_length": 8192,
                        "expires_at": "2026-08-19T13:00:00Z",
                        "details": {
                            "parameter_size": "8.2B",
                            "quantization_level": "Q4_K_M",
                        },
                    }
                ]
            },
            request=http_request,
        )

    async with OllamaClient(
        settings(), transport=httpx.MockTransport(handler)
    ) as client:
        result = await client.running_models()

    running = result.models[0]
    assert running.identifier == "qwen3:8b"
    assert running.size_vram == 4_000
    assert running.context_length == 8192
    assert running.details is not None
    assert running.details.quantization_level == "Q4_K_M"
