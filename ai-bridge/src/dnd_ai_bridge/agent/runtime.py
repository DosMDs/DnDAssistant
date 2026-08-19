"""Bounded application-level model/tool orchestration."""

from __future__ import annotations

import logging
import secrets
from collections.abc import Sequence

from ..errors import OneCProtocolError, OneCTransportError
from ..models import ChatMessage, ChatRole, ModelRequest, ModelToolCall, ToolDefinition
from ..onec_client import OneCClient
from ..provider import ModelProvider
from ..tool_registry import ToolRegistry
from .errors import (
    EmptyFinalResponseError,
    IterationLimitError,
    ToolCallLimitError,
    ToolNotAllowedError,
    ToolTransportFailureError,
    UnknownToolError,
)
from .models import AgentLimits, AgentResult
from .serialization import serialize_tool_result

logger = logging.getLogger(__name__)


class AgentRuntime:
    """Run a transient conversation while preserving provider and 1C boundaries."""

    def __init__(
        self,
        provider: ModelProvider,
        onec_client: OneCClient,
        *,
        tool_registry: ToolRegistry | None = None,
        limits: AgentLimits | None = None,
    ) -> None:
        self._provider = provider
        self._onec_client = onec_client
        self._tool_registry = tool_registry or ToolRegistry(onec_client)
        self._limits = limits or AgentLimits()

    async def run(self, messages: Sequence[ChatMessage]) -> AgentResult:
        """Execute one bounded run using a fresh dynamic tool snapshot."""

        run_id = secrets.token_hex(4)
        conversation = [message.model_copy(deep=True) for message in messages]
        tools = await self._tool_registry.load_tools()
        tools_by_name = {tool.name: tool for tool in tools}
        iterations = 0
        total_tool_calls = 0

        logger.info("[%s] agent run started tools=%d", run_id, len(tools))

        while True:
            if iterations >= self._limits.max_iterations:
                logger.warning(
                    "[%s] agent run stopped category=iteration_limit iterations=%d tools=%d",
                    run_id,
                    iterations,
                    total_tool_calls,
                )
                raise IterationLimitError(self._limits.max_iterations)

            iteration = iterations + 1
            logger.info("[%s] model completion iteration=%d", run_id, iteration)
            response = await self._provider.complete(
                ModelRequest(messages=conversation, tools=tools)
            )
            iterations = iteration
            assistant_message = response.message.model_copy(deep=True)
            conversation.append(assistant_message)

            if not assistant_message.tool_calls:
                if not assistant_message.content.strip():
                    logger.warning(
                        "[%s] agent run stopped category=empty_final_response iteration=%d",
                        run_id,
                        iteration,
                    )
                    raise EmptyFinalResponseError()

                logger.info(
                    "[%s] agent run completed iterations=%d tools=%d",
                    run_id,
                    iterations,
                    total_tool_calls,
                )
                return AgentResult(
                    final_message=assistant_message,
                    messages=conversation,
                    iterations=iterations,
                    total_tool_calls=total_tool_calls,
                )

            self._validate_tool_calls(assistant_message.tool_calls, tools_by_name)

            for tool_call in assistant_message.tool_calls:
                if total_tool_calls >= self._limits.max_total_tool_calls:
                    logger.warning(
                        "[%s] agent run stopped category=tool_call_limit "
                        "iteration=%d tools=%d",
                        run_id,
                        iteration,
                        total_tool_calls,
                    )
                    raise ToolCallLimitError(self._limits.max_total_tool_calls)

                logger.info(
                    "[%s] tool call iteration=%d tool=%s",
                    run_id,
                    iteration,
                    tool_call.name,
                )
                try:
                    tool_result = await self._onec_client.call_tool(
                        tool_call.name,
                        tool_call.arguments,
                    )
                except (OneCTransportError, OneCProtocolError) as exc:
                    logger.error(
                        "[%s] tool call failed category=tool_transport_failure tool=%s",
                        run_id,
                        tool_call.name,
                    )
                    raise ToolTransportFailureError(tool_call.name) from exc

                total_tool_calls += 1
                conversation.append(
                    ChatMessage(
                        role=ChatRole.TOOL,
                        tool_name=tool_call.name,
                        content=serialize_tool_result(tool_result),
                    )
                )
                logger.info(
                    "[%s] tool call completed tool=%s success=%s",
                    run_id,
                    tool_call.name,
                    tool_result.success,
                )

    @staticmethod
    def _validate_tool_calls(
        tool_calls: Sequence[ModelToolCall],
        tools_by_name: dict[str, ToolDefinition],
    ) -> None:
        """Validate the whole batch before any 1C call is attempted."""

        for tool_call in tool_calls:
            definition = tools_by_name.get(tool_call.name)
            if definition is None:
                raise UnknownToolError(tool_call.name)
            if not definition.read_only:
                raise ToolNotAllowedError(tool_call.name)
