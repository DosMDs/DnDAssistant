"""Provider-neutral boundary for a future model adapter."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from .models import ModelRequest, ModelResponse, ModelStreamChunk


class ModelProvider(Protocol):
    async def complete(self, request: ModelRequest) -> ModelResponse:
        """Complete a neutral model request."""

        ...

    def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamChunk]:
        """Stream one neutral model completion without executing tools."""

        ...
