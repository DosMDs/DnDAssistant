from __future__ import annotations

import asyncio

import pytest

from dnd_ai_bridge.agent import AgentResult
from dnd_ai_bridge.models import ChatMessage, ChatRole
from dnd_ai_bridge.service import AssistantService


class FakeRuntime:
    def __init__(self, *, cancel: bool = False) -> None:
        self.cancel = cancel
        self.calls: list[list[ChatMessage]] = []

    async def run(self, messages: list[ChatMessage]) -> AgentResult:
        self.calls.append(list(messages))
        if self.cancel:
            raise asyncio.CancelledError
        final = ChatMessage(role=ChatRole.ASSISTANT, content="Торвальд найден.")
        return AgentResult(
            final_message=final,
            messages=[*messages, final],
            iterations=1,
            total_tool_calls=0,
        )


async def test_service_returns_final_content_and_invokes_runtime_once() -> None:
    runtime = FakeRuntime()
    service = AssistantService(runtime)
    messages = [ChatMessage(role=ChatRole.USER, content="Где Торвальд?")]

    response = await service.run(messages)

    assert response == "Торвальд найден."
    assert runtime.calls == [messages]


async def test_service_does_not_convert_cancellation() -> None:
    runtime = FakeRuntime(cancel=True)
    service = AssistantService(runtime)

    with pytest.raises(asyncio.CancelledError):
        await service.run([ChatMessage(role=ChatRole.USER, content="Стоп")])

    assert len(runtime.calls) == 1
