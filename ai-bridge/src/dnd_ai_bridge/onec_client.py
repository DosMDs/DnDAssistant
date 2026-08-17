"""Async typed client for the 1C Assistant HTTP API."""

from __future__ import annotations

import logging
import secrets
from collections.abc import Mapping
from typing import Any
from urllib.parse import quote

import httpx
from pydantic import ValidationError

from .config import BridgeSettings
from .errors import (
    OneCAuthenticationError,
    OneCProtocolError,
    OneCTransportError,
)
from .models import (
    HealthResponse,
    ToolDescriptor,
    ToolResult,
    ToolsResponse,
    TransportErrorResponse,
)

logger = logging.getLogger(__name__)

_TRANSPORT_ERROR_STATUSES = {400, 413, 415, 500}
_MAX_SERVER_MESSAGE_LENGTH = 500


class OneCClient:
    """One-request-at-a-time async API client without retries or tool whitelist."""

    def __init__(
        self,
        settings: BridgeSettings,
        *,
        auth: httpx.Auth | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._settings = settings
        resolved_auth = (
            auth
            if auth is not None
            else httpx.BasicAuth(
                settings.onec_username,
                settings.onec_password.get_secret_value(),
            )
        )
        self._client = httpx.AsyncClient(
            base_url=f"{settings.onec_base_url}/",
            auth=resolved_auth,
            timeout=httpx.Timeout(settings.onec_timeout_seconds),
            transport=transport,
        )

    async def __aenter__(self) -> OneCClient:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    async def close(self) -> None:
        await self._client.aclose()

    async def health(self) -> HealthResponse:
        response = await self._request("GET", "health")
        payload = self._parse_success_json(response, endpoint="health")
        try:
            health = HealthResponse.model_validate(payload)
        except ValidationError as exc:
            raise OneCProtocolError("Invalid /health response contract") from exc

        if health.status != "ok":
            raise OneCProtocolError(
                f"Unexpected /health status: {health.status!r}"
            )
        if health.api_version != "1":
            raise OneCProtocolError(
                f"Unsupported 1C API version: {health.api_version!r}"
            )
        return health

    async def list_tools(self) -> list[ToolDescriptor]:
        response = await self._request("GET", "tools")
        payload = self._parse_success_json(response, endpoint="tools")
        try:
            envelope = ToolsResponse.model_validate(payload)
        except ValidationError as exc:
            raise OneCProtocolError("Invalid /tools response contract") from exc

        if not envelope.success:
            raise OneCProtocolError("The /tools endpoint returned success=false")
        if envelope.data is None:
            raise OneCProtocolError("The /tools response is missing data.tools")
        return envelope.data.tools

    async def call_tool(
        self,
        name: str,
        arguments: Mapping[str, Any] | None = None,
    ) -> ToolResult:
        request_payload = {} if arguments is None else dict(arguments)
        encoded_name = quote(name, safe="")
        response = await self._request(
            "POST",
            f"tools/{encoded_name}",
            json=request_payload,
            tool_name=name,
        )
        payload = self._parse_success_json(response, endpoint="tool call")
        try:
            return ToolResult.model_validate(payload)
        except ValidationError as exc:
            raise OneCProtocolError("Invalid tool result contract") from exc

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: Any | None = None,
        tool_name: str | None = None,
    ) -> httpx.Response:
        request_id = secrets.token_hex(4)
        label = f"tool={tool_name}" if tool_name is not None else f"/{path}"
        try:
            response = await self._client.request(method, path, json=json)
        except httpx.TimeoutException as exc:
            logger.error("[%s] %s %s timed out", request_id, method, label)
            raise OneCTransportError(
                "The 1C API request timed out",
                code="timeout",
            ) from exc
        except httpx.RequestError as exc:
            logger.error("[%s] %s %s transport failure", request_id, method, label)
            raise OneCTransportError(
                "Could not reach the 1C API",
                code="connection_error",
            ) from exc

        logger.info(
            "[%s] %s %s -> %d",
            request_id,
            method,
            label,
            response.status_code,
        )
        self._validate_http_status(response)
        return response

    @staticmethod
    def _parse_success_json(response: httpx.Response, *, endpoint: str) -> Any:
        try:
            return response.json()
        except ValueError as exc:
            raise OneCProtocolError(f"The {endpoint} response is not valid JSON") from exc

    @classmethod
    def _validate_http_status(cls, response: httpx.Response) -> None:
        status = response.status_code
        if status == 200:
            return
        if status in {401, 403}:
            code = "authentication_failed" if status == 401 else "authorization_failed"
            message = (
                "1C publication authentication failed"
                if status == 401
                else "1C publication denied access"
            )
            raise OneCAuthenticationError(
                message,
                code=code,
                http_status=status,
            )
        if status in _TRANSPORT_ERROR_STATUSES:
            code, message = cls._safe_transport_error(response)
            raise OneCTransportError(message, code=code, http_status=status)
        raise OneCTransportError(
            f"Unexpected HTTP status from the 1C API: {status}",
            code="unexpected_http_status",
            http_status=status,
        )

    @staticmethod
    def _safe_transport_error(response: httpx.Response) -> tuple[str, str]:
        try:
            envelope = TransportErrorResponse.model_validate(response.json())
        except (ValueError, ValidationError):
            return (
                "invalid_transport_error",
                f"1C API returned HTTP {response.status_code} with an invalid error response",
            )

        message = " ".join(envelope.error.message.split())
        if len(message) > _MAX_SERVER_MESSAGE_LENGTH:
            message = f"{message[:_MAX_SERVER_MESSAGE_LENGTH]}…"
        return envelope.error.code, message
