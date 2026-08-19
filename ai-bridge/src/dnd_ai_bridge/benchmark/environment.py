"""Portable environment and Ollama model metadata collection."""

from __future__ import annotations

import platform
from typing import Any

from ..ollama_client import OllamaClient
from ..ollama_models import OllamaRunningModel, OllamaShowModelResponse
from .models import EnvironmentMetadata, ModelMetadata


def local_environment(ollama_version: str | None) -> EnvironmentMetadata:
    return EnvironmentMetadata(
        os=platform.system(),
        os_version=platform.version(),
        architecture=platform.machine(),
        python_version=platform.python_version(),
        ollama_version=ollama_version,
    )


def _context_metadata(model_info: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in model_info.items()
        if "context" in key.casefold()
    }


def model_metadata(
    requested_name: str,
    shown: OllamaShowModelResponse | None,
    running: OllamaRunningModel | None,
) -> ModelMetadata:
    details = shown.details if shown is not None else None
    shown_name = None if shown is None else shown.model or shown.name
    shown_digest = None if shown is None else shown.digest
    return ModelMetadata(
        requested_name=requested_name,
        reported_name=(
            shown_name if running is None else running.identifier
        ),
        digest=(
            shown_digest if running is None or running.digest is None else running.digest
        ),
        parameter_size=None if details is None else details.parameter_size,
        quantization=None if details is None else details.quantization_level,
        capabilities=[] if shown is None else shown.capabilities,
        context_metadata=(
            {} if shown is None else _context_metadata(shown.model_info)
        ),
        allocated_context_length=(
            None if running is None else running.context_length
        ),
        size_vram=None if running is None else running.size_vram,
    )


async def collect_static_metadata(
    client: OllamaClient, model: str
) -> tuple[EnvironmentMetadata, OllamaShowModelResponse]:
    version = await client.version()
    shown = await client.show_model(model, verbose=True)
    return local_environment(version.version), shown
