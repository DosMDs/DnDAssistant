from __future__ import annotations

import logging

import pytest
from pydantic import ValidationError

from dnd_ai_bridge.config import BridgeSettings


def test_settings_load_from_environment_and_normalize_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DND_ONEC_BASE_URL", "http://127.0.0.1/demo/hs/assistant/v1/")
    monkeypatch.setenv("DND_ONEC_USERNAME", "assistant")
    monkeypatch.setenv("DND_ONEC_PASSWORD", "совершенно-секретно")
    monkeypatch.setenv("DND_ONEC_TIMEOUT_SECONDS", "3.5")

    settings = BridgeSettings(_env_file=None)

    assert settings.onec_base_url == "http://127.0.0.1/demo/hs/assistant/v1"
    assert settings.onec_username == "assistant"
    assert settings.onec_password.get_secret_value() == "совершенно-секретно"
    assert settings.onec_timeout_seconds == 3.5


def test_password_is_absent_from_settings_repr() -> None:
    password = "never-show-this-password"
    settings = BridgeSettings(
        onec_base_url="http://localhost/demo/hs/assistant/v1",
        onec_username="assistant",
        onec_password=password,
    )

    assert password not in repr(settings)
    assert "onec_password" not in repr(settings)


def test_timeout_defaults_to_ten_seconds() -> None:
    settings = BridgeSettings(
        onec_base_url="http://[::1]/demo/hs/assistant/v1/",
        onec_username="assistant",
        onec_password="secret",
    )

    assert settings.onec_timeout_seconds == 10.0
    assert settings.onec_base_url == "http://[::1]/demo/hs/assistant/v1"


def test_external_host_is_allowed_with_warning(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        settings = BridgeSettings(
            onec_base_url="https://onec.example.test/assistant/v1",
            onec_username="assistant",
            onec_password="secret",
        )

    assert settings.onec_base_url.startswith("https://")
    assert "not local" in caplog.text
    assert "secret" not in caplog.text


@pytest.mark.parametrize(
    "url",
    [
        "onec.example.test/assistant/v1",
        "ftp://localhost/assistant/v1",
        "http://localhost/assistant/v1?password=bad",
    ],
)
def test_invalid_base_urls_are_rejected(url: str) -> None:
    with pytest.raises(ValidationError):
        BridgeSettings(
            onec_base_url=url,
            onec_username="assistant",
            onec_password="secret",
        )


def test_non_positive_timeout_is_rejected() -> None:
    with pytest.raises(ValidationError):
        BridgeSettings(
            onec_base_url="http://localhost/assistant/v1",
            onec_username="assistant",
            onec_password="secret",
            onec_timeout_seconds=0,
        )

