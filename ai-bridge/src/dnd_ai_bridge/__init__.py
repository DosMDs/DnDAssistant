"""Public API for the D&D Assistant local AI bridge."""

from .config import BridgeSettings, OllamaSettings
from .errors import (
    BridgeError,
    OllamaConnectionError,
    OllamaError,
    OllamaHTTPStatusError,
    OllamaProtocolError,
    OllamaTimeoutError,
    OllamaTransportError,
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
    ModelPerformanceMetrics,
    ModelStreamChunk,
    ModelToolCall,
    ModelUsageMetrics,
    ToolDefinition,
    ToolDescriptor,
    ToolError,
    ToolResult,
)
from .onec_client import OneCClient
from .ollama_client import OllamaClient
from .ollama_provider import OllamaProvider
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
    "ModelPerformanceMetrics",
    "ModelRequest",
    "ModelResponse",
    "ModelStreamChunk",
    "ModelToolCall",
    "ModelUsageMetrics",
    "OllamaClient",
    "OllamaConnectionError",
    "OllamaError",
    "OllamaHTTPStatusError",
    "OllamaProtocolError",
    "OllamaProvider",
    "OllamaSettings",
    "OllamaTimeoutError",
    "OllamaTransportError",
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
