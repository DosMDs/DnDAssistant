"""Structured application-level agent failures."""

from __future__ import annotations

from ..errors import BridgeError


class AgentError(BridgeError):
    """Base class for expected orchestration failures."""

    def __init__(self, message: str, *, code: str) -> None:
        self.code = code
        super().__init__(message)


class UnknownToolError(AgentError):
    """The model requested a name absent from this run's registry."""

    def __init__(self, tool_name: str) -> None:
        self.tool_name = tool_name
        super().__init__(
            f"The model requested an unknown tool: {tool_name}",
            code="unknown_tool",
        )


class ToolNotAllowedError(AgentError):
    """The current execution policy rejected a known tool."""

    def __init__(self, tool_name: str) -> None:
        self.tool_name = tool_name
        super().__init__(
            f"The tool is not allowed by the read-only agent policy: {tool_name}",
            code="tool_not_allowed",
        )


class AgentLimitError(AgentError):
    """Base class for a hard run limit being reached."""


class IterationLimitError(AgentLimitError):
    """Starting another model completion would exceed the iteration limit."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        super().__init__(
            f"The agent iteration limit was reached ({limit})",
            code="iteration_limit",
        )


class ToolCallLimitError(AgentLimitError):
    """Executing another tool would exceed the total tool-call limit."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        super().__init__(
            f"The agent tool-call limit was reached ({limit})",
            code="tool_call_limit",
        )


class EmptyFinalResponseError(AgentError):
    """A terminal model response had no visible content."""

    def __init__(self) -> None:
        super().__init__(
            "The model returned an empty final response",
            code="empty_final_response",
        )


class ToolTransportFailureError(AgentError):
    """The 1C boundary failed before returning a normal ToolResult."""

    def __init__(self, tool_name: str) -> None:
        self.tool_name = tool_name
        super().__init__(
            f"The tool call failed at the 1C boundary: {tool_name}",
            code="tool_transport_failure",
        )
