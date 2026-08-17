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

