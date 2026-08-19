"""FastAPI application exposing the transient assistant use case."""

from __future__ import annotations

import asyncio
import logging
import secrets
import time
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from contextvars import ContextVar

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from ..agent import AgentError
from ..composition import ApplicationResources, application_resources_lifespan
from ..service import AssistantService
from .errors import runtime_error_status
from .models import (
    AgentRunRequest,
    AgentRunResponse,
    ErrorDetail,
    ErrorResponse,
    HealthResponse,
)

logger = logging.getLogger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"
request_id_context: ContextVar[str | None] = ContextVar("request_id", default=None)
ResourceLifespanFactory = Callable[
    [], AbstractAsyncContextManager[ApplicationResources]
]


class RequestContextMiddleware:
    """Attach correlation data without changing cancellation semantics."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope)
        request_id = request.headers.get(REQUEST_ID_HEADER) or secrets.token_hex(16)
        scope.setdefault("state", {})["request_id"] = request_id
        token = request_id_context.set(request_id)
        started = time.perf_counter()
        status_code = 500

        async def send_with_request_id(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                headers = MutableHeaders(scope=message)
                headers[REQUEST_ID_HEADER] = request_id
            await send(message)

        try:
            await self.app(scope, receive, send_with_request_id)
        except asyncio.CancelledError:
            raise
        finally:
            duration_ms = (time.perf_counter() - started) * 1000
            logger.info(
                "request_id=%s method=%s path=%s status=%d duration_ms=%.3f",
                request_id,
                request.method,
                request.url.path,
                status_code,
                duration_ms,
            )
            request_id_context.reset(token)


def _error_response(
    *, request_id: str, code: str, message: str, status_code: int
) -> JSONResponse:
    body = ErrorResponse(
        error=ErrorDetail(code=code, message=message),
        request_id=request_id,
    )
    return JSONResponse(
        status_code=status_code,
        content=body.model_dump(mode="json"),
        headers={REQUEST_ID_HEADER: request_id},
    )


def _request_id(request: Request) -> str:
    return request.state.request_id


def create_app(
    *, resource_lifespan: ResourceLifespanFactory | None = None
) -> FastAPI:
    """Create an ASGI app with injectable process-lifetime dependencies."""

    resources_context = resource_lifespan or application_resources_lifespan

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with resources_context() as resources:
            app.state.assistant_service = resources.assistant_service
            yield

    app = FastAPI(title="D&D Assistant AI bridge", version="1", lifespan=lifespan)
    app.add_middleware(RequestContextMiddleware)

    @app.exception_handler(RequestValidationError)
    async def invalid_request_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        del exc
        return _error_response(
            request_id=_request_id(request),
            code="invalid_request",
            message="Invalid request",
            status_code=422,
        )

    @app.exception_handler(AgentError)
    async def agent_error_handler(request: Request, exc: AgentError) -> JSONResponse:
        logger.warning("request_id=%s agent_error=%s", _request_id(request), exc.code)
        return _error_response(
            request_id=_request_id(request),
            code=exc.code,
            message=str(exc),
            status_code=runtime_error_status(exc),
        )

    @app.exception_handler(Exception)
    async def unexpected_error_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception(
            "request_id=%s unexpected request failure", _request_id(request)
        )
        return _error_response(
            request_id=_request_id(request),
            code="internal_error",
            message="Internal server error",
            status_code=500,
        )

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse()

    @app.post("/v1/agent/run", response_model=AgentRunResponse)
    async def run_agent(request: Request, payload: AgentRunRequest) -> AgentRunResponse:
        service: AssistantService = request.app.state.assistant_service
        response = await service.run(payload.messages)
        return AgentRunResponse(response=response, request_id=_request_id(request))

    return app


app = create_app()
