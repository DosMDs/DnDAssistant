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
from .provider import ModelProvider

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
    "ToolResult",
]

