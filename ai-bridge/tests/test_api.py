from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import AsyncIterator

import httpx
import pytest

from dnd_ai_bridge.agent import (
    AgentResult,
    EmptyFinalResponseError,
    IterationLimitError,
    ToolCallLimitError,
    ToolNotAllowedError,
    ToolTransportFailureError,
    UnknownToolError,
)
from dnd_ai_bridge.api.app import REQUEST_ID_HEADER, create_app
from dnd_ai_bridge.models import ChatMessage, ChatRole
from dnd_ai_bridge.service import AssistantService


class FakeService:
    def __init__(
        self,
        *,
        response: str = "Торвальд найден.",
        error: BaseException | None = None,
    ) -> None:
        self.response = response
        self.error = error
        self.calls: list[list[object]] = []

    async def run(self, messages: list[object]) -> str:
        self.calls.append(messages)
        if self.error is not None:
            raise self.error
        return self.response


class CountingRuntime:
    def __init__(self) -> None:
        self.calls = 0

    async def run(self, messages: list[ChatMessage]) -> AgentResult:
        self.calls += 1
        final = ChatMessage(role=ChatRole.ASSISTANT, content="Готово")
        return AgentResult(
            final_message=final,
            messages=[*messages, final],
            iterations=1,
            total_tool_calls=0,
        )


def _app_for(service: object, events: list[str] | None = None):
    @asynccontextmanager
    async def resources():  # type: ignore[no-untyped-def]
        if events is not None:
            events.append("startup")
        try:
            yield SimpleNamespace(assistant_service=service)
        finally:
            if events is not None:
                events.append("shutdown")

    return create_app(resource_lifespan=resources)


@asynccontextmanager
async def _client(
    service: object,
    *,
    raise_app_exceptions: bool = True,
    events: list[str] | None = None,
) -> AsyncIterator[httpx.AsyncClient]:
    app = _app_for(service, events)
    transport = httpx.ASGITransport(
        app=app,
        raise_app_exceptions=raise_app_exceptions,
    )
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            yield client


async def test_health_is_cheap_and_reports_ok() -> None:
    service = FakeService()

    async with _client(service) as client:
        response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert service.calls == []


async def test_successful_agent_call_generates_request_id_and_invokes_once() -> None:
    service = FakeService()

    async with _client(service) as client:
        response = await client.post(
            "/v1/agent/run",
            json={"messages": [{"role": "user", "content": "Где Торвальд?"}]},
        )

    body = response.json()
    assert response.status_code == 200
    assert body["response"] == "Торвальд найден."
    assert body["request_id"]
    assert response.headers[REQUEST_ID_HEADER] == body["request_id"]
    assert len(service.calls) == 1
    assert service.calls[0][0].content == "Где Торвальд?"


async def test_request_id_is_propagated_to_header_and_body() -> None:
    service = FakeService()

    async with _client(service) as client:
        response = await client.post(
            "/v1/agent/run",
            headers={REQUEST_ID_HEADER: "abc-123"},
            json={"messages": [{"role": "user", "content": "Где Торвальд?"}]},
        )

    assert response.json()["request_id"] == "abc-123"
    assert response.headers[REQUEST_ID_HEADER] == "abc-123"


async def test_one_http_request_crosses_service_and_runtime_exactly_once() -> None:
    runtime = CountingRuntime()
    service = AssistantService(runtime)

    async with _client(service) as client:
        response = await client.post(
            "/v1/agent/run",
            json={"messages": [{"role": "user", "content": "Продолжай"}]},
        )

    assert response.status_code == 200
    assert response.json()["response"] == "Готово"
    assert runtime.calls == 1


async def test_invalid_request_has_predictable_safe_envelope() -> None:
    service = FakeService()

    async with _client(service) as client:
        response = await client.post(
            "/v1/agent/run",
            json={"messages": [{"role": "not-a-role", "content": "x"}]},
        )

    body = response.json()
    assert response.status_code == 422
    assert body["error"] == {
        "code": "invalid_request",
        "message": "Invalid request",
    }
    assert body["request_id"] == response.headers[REQUEST_ID_HEADER]
    assert service.calls == []


@pytest.mark.parametrize(
    ("error", "expected_status"),
    [
        (UnknownToolError("missing"), 502),
        (ToolNotAllowedError("write_tool"), 502),
        (IterationLimitError(8), 422),
        (ToolCallLimitError(16), 422),
        (EmptyFinalResponseError(), 502),
        (ToolTransportFailureError("search_entities"), 503),
    ],
)
async def test_runtime_error_code_is_preserved(
    error: BaseException, expected_status: int
) -> None:
    service = FakeService(error=error)

    async with _client(service) as client:
        response = await client.post(
            "/v1/agent/run",
            json={"messages": [{"role": "user", "content": "x"}]},
        )

    body = response.json()
    assert response.status_code == expected_status
    assert body["error"]["code"] == error.code
    assert body["request_id"] == response.headers[REQUEST_ID_HEADER]
    assert len(service.calls) == 1


async def test_unexpected_exception_is_sanitized() -> None:
    sensitive = "C:\\private\\token.txt secret-password"
    service = FakeService(error=RuntimeError(sensitive))

    async with _client(service, raise_app_exceptions=False) as client:
        response = await client.post(
            "/v1/agent/run",
            json={"messages": [{"role": "user", "content": "x"}]},
        )

    body_text = response.text
    body = response.json()
    assert response.status_code == 500
    assert body["error"] == {
        "code": "internal_error",
        "message": "Internal server error",
    }
    assert body["request_id"] == response.headers[REQUEST_ID_HEADER]
    assert sensitive not in body_text
    assert "Traceback" not in body_text


async def test_cancellation_is_not_converted_to_internal_error() -> None:
    service = FakeService(error=asyncio.CancelledError())

    async with _client(service, raise_app_exceptions=False) as client:
        with pytest.raises(asyncio.CancelledError):
            await client.post(
                "/v1/agent/run",
                json={"messages": [{"role": "user", "content": "x"}]},
            )

    assert len(service.calls) == 1


async def test_injected_resources_follow_app_lifespan() -> None:
    events: list[str] = []
    service = FakeService()

    async with _client(service, events=events) as client:
        assert events == ["startup"]
        response = await client.get("/health")
        assert response.status_code == 200

    assert events == ["startup", "shutdown"]
