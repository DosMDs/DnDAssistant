"""Provider-neutral boundary for a future model adapter."""

from __future__ import annotations

from typing import Protocol

from .models import ModelRequest, ModelResponse


class ModelProvider(Protocol):
    async def complete(self, request: ModelRequest) -> ModelResponse:
        """Complete a neutral model request."""

        ...

