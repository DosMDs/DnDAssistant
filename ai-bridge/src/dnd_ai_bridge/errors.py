"""Bridge exception hierarchy."""

from __future__ import annotations


class BridgeError(Exception):
    """Base class for expected bridge failures."""


class OneCTransportError(BridgeError):
    """Network failure or a non-success HTTP transport response."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "transport_error",
        http_status: int | None = None,
    ) -> None:
        self.code = code
        self.http_status = http_status
        super().__init__(message)


class OneCAuthenticationError(OneCTransportError):
    """Publication authentication or authorization failure."""


class OneCProtocolError(BridgeError):
    """A HTTP 200 response that violates the expected API contract."""


class OllamaError(BridgeError):
    """Base class for expected local Ollama failures."""


class OllamaTransportError(OllamaError):
    """Base class for failures while communicating with Ollama."""

    def __init__(
        self,
        message: str,
        *,
        code: str,
        http_status: int | None = None,
    ) -> None:
        self.code = code
        self.http_status = http_status
        super().__init__(message)


class OllamaTimeoutError(OllamaTransportError):
    """The configured Ollama request timeout elapsed."""

    def __init__(self, message: str = "The Ollama request timed out") -> None:
        super().__init__(message, code="timeout")


class OllamaConnectionError(OllamaTransportError):
    """The configured Ollama endpoint could not be reached."""

    def __init__(self, message: str = "Could not reach Ollama") -> None:
        super().__init__(message, code="connection_error")


class OllamaHTTPStatusError(OllamaTransportError):
    """Ollama returned a non-success HTTP status."""

    def __init__(self, message: str, *, http_status: int) -> None:
        super().__init__(
            message,
            code="unexpected_http_status",
            http_status=http_status,
        )


class OllamaProtocolError(OllamaError):
    """Ollama returned malformed JSON or an invalid response contract."""
