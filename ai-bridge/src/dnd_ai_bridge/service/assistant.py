"""Application boundary for transient assistant requests."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from ..agent import AgentResult, AgentRuntime
from ..models import ChatMessage


class AgentRunner(Protocol):
    async def run(self, messages: Sequence[ChatMessage]) -> AgentResult:
        """Execute one transient agent run."""

        ...


class AssistantService:
    """Expose assistant use cases without depending on an HTTP framework."""

    def __init__(self, runtime: AgentRunner | AgentRuntime) -> None:
        self._runtime = runtime

    async def run(self, messages: Sequence[ChatMessage]) -> str:
        result = await self._runtime.run(messages)
        return result.final_message.content
