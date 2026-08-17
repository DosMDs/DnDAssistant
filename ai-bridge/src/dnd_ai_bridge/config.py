"""Environment-backed configuration for the bridge."""

from __future__ import annotations

import logging
from urllib.parse import urlsplit, urlunsplit

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


class BridgeSettings(BaseSettings):
    """Connection settings loaded from ``DND_*`` environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="DND_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    onec_base_url: str
    onec_username: str
    onec_password: SecretStr = Field(repr=False)
    onec_timeout_seconds: float = Field(default=10.0, gt=0)

    @field_validator("onec_base_url")
    @classmethod
    def normalize_base_url(cls, value: str) -> str:
        """Validate HTTP(S) and remove only trailing path slashes."""

        candidate = value.strip()
        parts = urlsplit(candidate)
        if parts.scheme not in {"http", "https"} or not parts.hostname:
            raise ValueError("onec_base_url must be an absolute HTTP(S) URL")
        if parts.query or parts.fragment:
            raise ValueError("onec_base_url must not contain a query or fragment")

        normalized_path = parts.path.rstrip("/")
        if not normalized_path:
            normalized_path = ""
        return urlunsplit((parts.scheme, parts.netloc, normalized_path, "", ""))

    @model_validator(mode="after")
    def warn_for_non_local_endpoint(self) -> BridgeSettings:
        host = urlsplit(self.onec_base_url).hostname
        if host is not None and host.lower() not in _LOCAL_HOSTS:
            logger.warning(
                "1C endpoint host is not local: %s; local hosts are preferred",
                host,
            )
        return self

