"""Async httpx transport for the local Ollama HTTP API."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from .config import OllamaSettings
from .errors import (
    OllamaConnectionError,
    OllamaHTTPStatusError,
    OllamaProtocolError,
    OllamaTimeoutError,
)
from .ollama_models import (
    OllamaChatRequest,
    OllamaChatResponse,
    OllamaErrorResponse,
    OllamaModelsResponse,
    OllamaShowModelResponse,
    OllamaVersionResponse,
)

ResponseT = TypeVar("ResponseT", bound=BaseModel)
_MAX_ERROR_LENGTH = 500


class OllamaClient:
    """One-request-at-a-time client with no auth, retries, or cloud fallback."""

    def __init__(
        self,
        settings: OllamaSettings | None = None,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        resolved = settings or OllamaSettings()
        self._client = httpx.AsyncClient(
            base_url=f"{resolved.ollama_base_url}/",
            timeout=httpx.Timeout(resolved.ollama_timeout_seconds),
            transport=transport,
            trust_env=False,
        )

    async def __aenter__(self) -> OllamaClient:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    async def close(self) -> None:
        await self._client.aclose()

    async def chat(self, request: OllamaChatRequest) -> OllamaChatResponse:
        """Perform one non-streaming chat completion."""

        payload = self._chat_payload(request, stream=False)
        response = await self._request_model(
            "POST",
            "api/chat",
            OllamaChatResponse,
            payload=payload,
            endpoint="/api/chat",
        )
        if not response.done:
            raise OllamaProtocolError(
                "The non-streaming /api/chat response is not final"
            )
        return response

    async def stream_chat(
        self, request: OllamaChatRequest
    ) -> AsyncIterator[OllamaChatResponse]:
        """Yield validated NDJSON chunks for one streaming completion."""

        payload = self._chat_payload(request, stream=True)
        saw_done = False
        try:
            async with self._client.stream(
                "POST",
                "api/chat",
                content=self._encode_json(payload),
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/x-ndjson",
                },
            ) as response:
                self._validate_status(response)
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        chunk = OllamaChatResponse.model_validate_json(line)
                    except (ValueError, ValidationError) as exc:
                        raise OllamaProtocolError(
                            "The /api/chat stream contains invalid NDJSON"
                        ) from exc
                    if saw_done:
                        raise OllamaProtocolError(
                            "The /api/chat stream contains data after the final chunk"
                        )
                    saw_done = chunk.done
                    yield chunk
        except httpx.TimeoutException as exc:
            raise OllamaTimeoutError() from exc
        except httpx.RequestError as exc:
            raise OllamaConnectionError() from exc

        if not saw_done:
            raise OllamaProtocolError(
                "The /api/chat stream ended without a final chunk"
            )

    async def version(self) -> OllamaVersionResponse:
        return await self._request_model(
            "GET", "api/version", OllamaVersionResponse, endpoint="/api/version"
        )

    async def list_models(self) -> OllamaModelsResponse:
        return await self._request_model(
            "GET", "api/tags", OllamaModelsResponse, endpoint="/api/tags"
        )

    async def show_model(
        self, model: str, *, verbose: bool = False
    ) -> OllamaShowModelResponse:
        return await self._request_model(
            "POST",
            "api/show",
            OllamaShowModelResponse,
            payload={"model": model, "verbose": verbose},
            endpoint="/api/show",
        )

    async def _request_model(
        self,
        method: str,
        path: str,
        response_type: type[ResponseT],
        *,
        payload: dict[str, Any] | None = None,
        endpoint: str,
    ) -> ResponseT:
        try:
            response = await self._client.request(
                method,
                path,
                content=None if payload is None else self._encode_json(payload),
                headers={"Accept": "application/json"}
                if payload is None
                else {"Content-Type": "application/json", "Accept": "application/json"},
            )
        except httpx.TimeoutException as exc:
            raise OllamaTimeoutError() from exc
        except httpx.RequestError as exc:
            raise OllamaConnectionError() from exc

        self._validate_status(response)
        try:
            return response_type.model_validate_json(response.content)
        except (ValueError, ValidationError) as exc:
            raise OllamaProtocolError(
                f"The {endpoint} response is not valid JSON for its contract"
            ) from exc

    @staticmethod
    def _chat_payload(
        request: OllamaChatRequest, *, stream: bool
    ) -> dict[str, Any]:
        payload = request.model_dump(
            mode="json", exclude_none=True, exclude_defaults=True
        )
        payload["stream"] = stream
        payload["think"] = False
        return payload

    @staticmethod
    def _encode_json(payload: dict[str, Any]) -> bytes:
        return json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")

    @staticmethod
    def _validate_status(response: httpx.Response) -> None:
        if response.status_code == 200:
            return

        message = f"Ollama returned HTTP {response.status_code}"
        try:
            parsed = OllamaErrorResponse.model_validate_json(response.content)
        except (ValueError, ValidationError):
            pass
        else:
            cleaned = " ".join(parsed.error.split())
            if cleaned:
                if len(cleaned) > _MAX_ERROR_LENGTH:
                    cleaned = f"{cleaned[:_MAX_ERROR_LENGTH]}…"
                message = cleaned
        raise OllamaHTTPStatusError(message, http_status=response.status_code)
