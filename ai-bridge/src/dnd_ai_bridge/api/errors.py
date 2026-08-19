"""Stable mapping from application failures to HTTP semantics."""

from __future__ import annotations

from fastapi import status

from ..agent import AgentError

_RUNTIME_STATUS_BY_CODE = {
    "unknown_tool": status.HTTP_502_BAD_GATEWAY,
    "tool_not_allowed": status.HTTP_502_BAD_GATEWAY,
    "iteration_limit": status.HTTP_422_UNPROCESSABLE_CONTENT,
    "tool_call_limit": status.HTTP_422_UNPROCESSABLE_CONTENT,
    "empty_final_response": status.HTTP_502_BAD_GATEWAY,
    "tool_transport_failure": status.HTTP_503_SERVICE_UNAVAILABLE,
}


def runtime_error_status(error: AgentError) -> int:
    """Return the transport status without changing the application code."""

    return _RUNTIME_STATUS_BY_CODE.get(error.code, status.HTTP_500_INTERNAL_SERVER_ERROR)
