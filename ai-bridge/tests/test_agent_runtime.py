from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import pytest
from pydantic import ValidationError

from dnd_ai_bridge.agent import (
    AgentLimits,
    AgentRuntime,
    EmptyFinalResponseError,
    IterationLimitError,
    ToolCallLimitError,
    ToolNotAllowedError,
    ToolTransportFailureError,
    UnknownToolError,
    serialize_tool_result,
)
from dnd_ai_bridge.errors import OneCTransportError
from dnd_ai_bridge.models import (
    ChatMessage,
    ChatRole,
    ModelRequest,
    ModelResponse,
    ModelStreamChunk,
    ModelToolCall,
    ToolDefinition,
    ToolError,
    ToolResult,
)


def tool_definition(name: str, *, read_only: bool = True) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description=f"Tool {name}",
        read_only=read_only,
        parameters={"type": "object", "properties": {}},
    )


def assistant(
    content: str = "",
    *tool_calls: tuple[str, dict[str, Any]],
) -> ModelResponse:
    return ModelResponse(
        message=ChatMessage(
            role=ChatRole.ASSISTANT,
            content=content,
            tool_calls=[
                ModelToolCall(name=name, arguments=arguments)
                for name, arguments in tool_calls
            ],
        )
    )


class FakeProvider:
    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = list(responses)
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request.model_copy(deep=True))
        return self.responses.pop(0)

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamChunk]:
        if False:
            yield ModelStreamChunk()


class FakeRegistry:
    def __init__(self, snapshots: list[list[ToolDefinition]]) -> None:
        self.snapshots = list(snapshots)
        self.load_calls = 0

    async def load_tools(self) -> list[ToolDefinition]:
        snapshot = self.snapshots[self.load_calls]
        self.load_calls += 1
        return [tool.model_copy(deep=True) for tool in snapshot]


class FakeOneCClient:
    def __init__(
        self,
        results: list[ToolResult | BaseException],
        *,
        yield_during_call: bool = False,
    ) -> None:
        self.results = list(results)
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.yield_during_call = yield_during_call
        self.active_calls = 0
        self.max_active_calls = 0

    async def call_tool(
        self, name: str, arguments: dict[str, Any]
    ) -> ToolResult:
        self.calls.append((name, dict(arguments)))
        self.active_calls += 1
        self.max_active_calls = max(self.max_active_calls, self.active_calls)
        try:
            if self.yield_during_call:
                await asyncio.sleep(0)
            result = self.results.pop(0)
            if isinstance(result, BaseException):
                raise result
            return result
        finally:
            self.active_calls -= 1


def runtime(
    provider: FakeProvider,
    client: FakeOneCClient,
    tools: list[ToolDefinition],
    *,
    limits: AgentLimits | None = None,
    registry: FakeRegistry | None = None,
) -> tuple[AgentRuntime, FakeRegistry]:
    selected_registry = registry or FakeRegistry([tools])
    agent = AgentRuntime(
        provider,
        client,  # type: ignore[arg-type]
        tool_registry=selected_registry,  # type: ignore[arg-type]
        limits=limits,
    )
    return agent, selected_registry


@pytest.mark.asyncio
async def test_no_tool_response_is_final_and_reports_transcript_counters() -> None:
    provider = FakeProvider([assistant("Готово")])
    client = FakeOneCClient([])
    agent, registry = runtime(provider, client, [tool_definition("current_tool")])
    initial = [ChatMessage(role=ChatRole.USER, content="Вопрос")]

    result = await agent.run(initial)

    assert result.final_message.content == "Готово"
    assert [message.role for message in result.messages] == [
        ChatRole.USER,
        ChatRole.ASSISTANT,
    ]
    assert result.iterations == 1
    assert result.total_tool_calls == 0
    assert registry.load_calls == 1
    assert [tool.name for tool in provider.requests[0].tools] == ["current_tool"]
    assert client.calls == []
    assert len(initial) == 1


@pytest.mark.asyncio
async def test_one_tool_result_is_serialized_and_followed_by_final_response() -> None:
    provider = FakeProvider(
        [
            assistant("Проверю", ("get_entity", {"id": "42"})),
            assistant("Торвальд найден"),
        ]
    )
    client = FakeOneCClient(
        [ToolResult(success=True, data={"name": "Торвальд"}, error=None)]
    )
    agent, _ = runtime(provider, client, [tool_definition("get_entity")])

    result = await agent.run(
        [ChatMessage(role=ChatRole.USER, content="Где Торвальд?")]
    )

    assert result.iterations == 2
    assert result.total_tool_calls == 1
    assert client.calls == [("get_entity", {"id": "42"})]
    assert [message.role for message in result.messages] == [
        ChatRole.USER,
        ChatRole.ASSISTANT,
        ChatRole.TOOL,
        ChatRole.ASSISTANT,
    ]
    tool_request = result.messages[1]
    assert tool_request.content == "Проверю"
    assert tool_request.tool_calls == [
        ModelToolCall(name="get_entity", arguments={"id": "42"})
    ]
    tool_message = result.messages[2]
    assert tool_message.tool_name == "get_entity"
    assert json.loads(tool_message.content) == {
        "success": True,
        "data": {"name": "Торвальд"},
        "error": None,
    }
    assert provider.requests[1].messages == result.messages[:3]


@pytest.mark.asyncio
async def test_multiple_tool_calls_execute_sequentially_in_model_order() -> None:
    provider = FakeProvider(
        [
            assistant(
                "",
                ("first", {"value": 1}),
                ("second", {"value": 2}),
            ),
            assistant("Завершено"),
        ]
    )
    client = FakeOneCClient(
        [
            ToolResult(success=True, data={"order": 1}, error=None),
            ToolResult(success=True, data={"order": 2}, error=None),
        ],
        yield_during_call=True,
    )
    agent, _ = runtime(
        provider,
        client,
        [tool_definition("first"), tool_definition("second")],
    )

    result = await agent.run([ChatMessage(role=ChatRole.USER, content="Два")])

    assert client.calls == [
        ("first", {"value": 1}),
        ("second", {"value": 2}),
    ]
    assert client.max_active_calls == 1
    assert [message.tool_name for message in result.messages if message.role == "tool"] == [
        "first",
        "second",
    ]


@pytest.mark.asyncio
async def test_registry_is_reloaded_for_each_run() -> None:
    provider = FakeProvider([assistant("Первый"), assistant("Второй")])
    client = FakeOneCClient([])
    registry = FakeRegistry(
        [[tool_definition("old")], [tool_definition("new")]]
    )
    agent, _ = runtime(provider, client, [], registry=registry)

    await agent.run([ChatMessage(role=ChatRole.USER, content="1")])
    await agent.run([ChatMessage(role=ChatRole.USER, content="2")])

    assert registry.load_calls == 2
    assert [tool.name for tool in provider.requests[0].tools] == ["old"]
    assert [tool.name for tool in provider.requests[1].tools] == ["new"]


@pytest.mark.asyncio
async def test_unknown_tool_is_rejected_before_any_onec_call() -> None:
    provider = FakeProvider(
        [assistant("", ("known", {}), ("hallucinated", {"secret": "value"}))]
    )
    client = FakeOneCClient(
        [ToolResult(success=True, data=None, error=None)]
    )
    agent, _ = runtime(provider, client, [tool_definition("known")])

    with pytest.raises(UnknownToolError) as caught:
        await agent.run([ChatMessage(role=ChatRole.USER, content="x")])

    assert caught.value.code == "unknown_tool"
    assert caught.value.tool_name == "hallucinated"
    assert "secret" not in str(caught.value)
    assert client.calls == []


@pytest.mark.asyncio
async def test_non_read_only_tool_is_rejected_before_any_onec_call() -> None:
    provider = FakeProvider([assistant("", ("write_journal", {"text": "secret"}))])
    client = FakeOneCClient([])
    agent, _ = runtime(
        provider,
        client,
        [tool_definition("write_journal", read_only=False)],
    )

    with pytest.raises(ToolNotAllowedError) as caught:
        await agent.run([ChatMessage(role=ChatRole.USER, content="x")])

    assert caught.value.code == "tool_not_allowed"
    assert "secret" not in str(caught.value)
    assert client.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("error_code", ["domain_failure", "invalid_arguments"])
async def test_unsuccessful_tool_result_is_a_normal_model_message(
    error_code: str,
) -> None:
    provider = FakeProvider(
        [assistant("", ("lookup", {"query": "x"})), assistant("Исправлено")]
    )
    client = FakeOneCClient(
        [
            ToolResult(
                success=False,
                data=None,
                error=ToolError(code=error_code, message="Ошибка инструмента"),
            )
        ]
    )
    agent, _ = runtime(provider, client, [tool_definition("lookup")])

    result = await agent.run([ChatMessage(role=ChatRole.USER, content="x")])

    payload = json.loads(provider.requests[1].messages[-1].content)
    assert payload == {
        "success": False,
        "data": None,
        "error": {"code": error_code, "message": "Ошибка инструмента"},
    }
    assert result.total_tool_calls == 1


@pytest.mark.asyncio
async def test_onec_failure_is_chained_without_retry() -> None:
    provider = FakeProvider([assistant("", ("lookup", {"private": "value"}))])
    original = OneCTransportError("server details", code="connection_error")
    client = FakeOneCClient([original])
    agent, _ = runtime(provider, client, [tool_definition("lookup")])

    with pytest.raises(ToolTransportFailureError) as caught:
        await agent.run([ChatMessage(role=ChatRole.USER, content="x")])

    assert caught.value.code == "tool_transport_failure"
    assert caught.value.__cause__ is original
    assert "private" not in str(caught.value)
    assert client.calls == [("lookup", {"private": "value"})]
    assert len(provider.requests) == 1


@pytest.mark.asyncio
async def test_content_with_tool_calls_is_not_treated_as_final() -> None:
    provider = FakeProvider(
        [assistant("Промежуточный текст", ("lookup", {})), assistant("Финал")]
    )
    client = FakeOneCClient(
        [ToolResult(success=True, data={"x": 1}, error=None)]
    )
    agent, _ = runtime(provider, client, [tool_definition("lookup")])

    result = await agent.run([ChatMessage(role=ChatRole.USER, content="x")])

    assert result.final_message.content == "Финал"
    assert len(provider.requests) == 2
    assert result.messages[1].content == "Промежуточный текст"


def test_agent_limits_require_positive_values() -> None:
    with pytest.raises(ValidationError):
        AgentLimits(max_iterations=0)
    with pytest.raises(ValidationError):
        AgentLimits(max_total_tool_calls=-1)


@pytest.mark.asyncio
async def test_iteration_limit_stops_before_next_model_completion() -> None:
    provider = FakeProvider([assistant("", ("lookup", {}))])
    client = FakeOneCClient(
        [ToolResult(success=True, data={"x": 1}, error=None)]
    )
    agent, _ = runtime(
        provider,
        client,
        [tool_definition("lookup")],
        limits=AgentLimits(max_iterations=1),
    )

    with pytest.raises(IterationLimitError) as caught:
        await agent.run([ChatMessage(role=ChatRole.USER, content="x")])

    assert caught.value.code == "iteration_limit"
    assert len(provider.requests) == 1
    assert len(client.calls) == 1


@pytest.mark.asyncio
async def test_tool_call_limit_stops_before_the_excess_call() -> None:
    provider = FakeProvider(
        [assistant("", ("first", {}), ("second", {}))]
    )
    client = FakeOneCClient(
        [ToolResult(success=True, data={"x": 1}, error=None)]
    )
    agent, _ = runtime(
        provider,
        client,
        [tool_definition("first"), tool_definition("second")],
        limits=AgentLimits(max_total_tool_calls=1),
    )

    with pytest.raises(ToolCallLimitError) as caught:
        await agent.run([ChatMessage(role=ChatRole.USER, content="x")])

    assert caught.value.code == "tool_call_limit"
    assert client.calls == [("first", {})]
    assert len(provider.requests) == 1


@pytest.mark.asyncio
async def test_empty_terminal_response_is_an_error() -> None:
    provider = FakeProvider([assistant(" \n ")])
    client = FakeOneCClient([])
    agent, _ = runtime(provider, client, [])

    with pytest.raises(EmptyFinalResponseError) as caught:
        await agent.run([ChatMessage(role=ChatRole.USER, content="x")])

    assert caught.value.code == "empty_final_response"
    assert client.calls == []


class BlockingProvider:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.calls = 0

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.calls += 1
        self.started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamChunk]:
        if False:
            yield ModelStreamChunk()


@pytest.mark.asyncio
async def test_cancellation_during_model_completion_propagates_immediately() -> None:
    provider = BlockingProvider()
    client = FakeOneCClient([])
    agent = AgentRuntime(
        provider,
        client,  # type: ignore[arg-type]
        tool_registry=FakeRegistry([[]]),  # type: ignore[arg-type]
    )
    task = asyncio.create_task(
        agent.run([ChatMessage(role=ChatRole.USER, content="x")])
    )
    await provider.started.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0)

    assert provider.calls == 1
    assert client.calls == []


class BlockingOneCClient(FakeOneCClient):
    def __init__(self) -> None:
        super().__init__([])
        self.started = asyncio.Event()

    async def call_tool(
        self, name: str, arguments: dict[str, Any]
    ) -> ToolResult:
        self.calls.append((name, dict(arguments)))
        self.started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


@pytest.mark.asyncio
async def test_cancellation_during_tool_execution_propagates_immediately() -> None:
    provider = FakeProvider(
        [assistant("", ("lookup", {})), assistant("must not run")]
    )
    client = BlockingOneCClient()
    agent, _ = runtime(provider, client, [tool_definition("lookup")])
    task = asyncio.create_task(
        agent.run([ChatMessage(role=ChatRole.USER, content="x")])
    )
    await client.started.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0)

    assert len(provider.requests) == 1
    assert client.calls == [("lookup", {})]


def test_tool_result_json_is_compact_deterministic_and_unicode() -> None:
    first = ToolResult(
        success=True,
        data={"name": "Торвальд", "nested": {"я": 2, "а": 1}},
        error=None,
    )
    second = ToolResult(
        success=True,
        data={"nested": {"а": 1, "я": 2}, "name": "Торвальд"},
        error=None,
    )

    encoded = serialize_tool_result(first)

    assert encoded == serialize_tool_result(second)
    assert encoded == (
        '{"success":true,"data":{"name":"Торвальд",'
        '"nested":{"а":1,"я":2}},"error":null}'
    )
    assert "\\u" not in encoded
