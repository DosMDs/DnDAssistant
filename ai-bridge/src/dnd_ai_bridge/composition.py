"""Composition root and lifecycle for the local assistant service."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator

from .agent import AgentRuntime
from .config import AgentSettings, BridgeSettings, OllamaSettings
from .ollama_client import OllamaClient
from .ollama_provider import OllamaProvider
from .onec_client import OneCClient
from .service import AssistantService
from .tool_registry import ToolRegistry


@dataclass(slots=True)
class ApplicationResources:
    """Long-lived reusable dependencies owned by one ASGI application."""

    assistant_service: AssistantService
    onec_client: OneCClient
    ollama_client: OllamaClient

    async def close(self) -> None:
        """Close both transports even if the first close operation fails."""

        try:
            await self.ollama_client.close()
        finally:
            await self.onec_client.close()


def build_application_resources(
    *,
    bridge_settings: BridgeSettings | None = None,
    ollama_settings: OllamaSettings | None = None,
    agent_settings: AgentSettings | None = None,
) -> ApplicationResources:
    """Build the complete dependency graph once for process-level reuse."""

    resolved_bridge = bridge_settings or BridgeSettings()
    resolved_ollama = ollama_settings or OllamaSettings()
    resolved_agent = agent_settings or AgentSettings()

    onec_client = OneCClient(resolved_bridge)
    ollama_client = OllamaClient(resolved_ollama)
    provider = OllamaProvider(ollama_client, resolved_agent.model)
    registry = ToolRegistry(onec_client)
    runtime = AgentRuntime(provider, onec_client, tool_registry=registry)
    return ApplicationResources(
        assistant_service=AssistantService(runtime),
        onec_client=onec_client,
        ollama_client=ollama_client,
    )


@asynccontextmanager
async def application_resources_lifespan(
    *,
    bridge_settings: BridgeSettings | None = None,
    ollama_settings: OllamaSettings | None = None,
    agent_settings: AgentSettings | None = None,
) -> AsyncIterator[ApplicationResources]:
    """Own reusable clients for exactly one ASGI application lifespan."""

    resources = build_application_resources(
        bridge_settings=bridge_settings,
        ollama_settings=ollama_settings,
        agent_settings=agent_settings,
    )
    try:
        yield resources
    finally:
        await resources.close()
