"""Public API for the D&D Assistant local AI bridge."""

from .config import BridgeSettings
from .errors import (
    BridgeError,
    OneCAuthenticationError,
    OneCProtocolError,
    OneCTransportError,
)
from .models import (
    ChatMessage,
    ChatRole,
    HealthResponse,
    ModelRequest,
    ModelResponse,
    ToolDefinition,
    ToolDescriptor,
    ToolError,
    ToolResult,
)
from .onec_client import OneCClient
from .ollama_tools import to_ollama_tools
from .provider import ModelProvider
from .tool_registry import ToolRegistry

__all__ = [
    "BridgeError",
    "BridgeSettings",
    "ChatMessage",
    "ChatRole",
    "HealthResponse",
    "ModelProvider",
    "ModelRequest",
    "ModelResponse",
    "OneCAuthenticationError",
    "OneCClient",
    "OneCProtocolError",
    "OneCTransportError",
    "ToolDefinition",
    "ToolDescriptor",
    "ToolError",
    "ToolRegistry",
    "ToolResult",
    "to_ollama_tools",
]
