from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from dnd_ai_bridge.cli import _parser, _print_json, main


def test_cli_json_keeps_russian_characters_readable(
    capsys: pytest.CaptureFixture[str],
) -> None:
    phrases = [
        "Торвальд Железнорукий",
        "5 Миртула 1492 ЛД",
        "Гильдия кузнецов",
    ]

    _print_json({"values": phrases})
    captured = capsys.readouterr()

    assert "Торвальд Железнорукий" in captured.out
    assert "\\u" not in captured.out
    assert json.loads(captured.out) == {"values": phrases}


def test_benchmark_cli_arguments_support_modes_and_selection() -> None:
    args = _parser().parse_args(
        [
            "benchmark",
            "run",
            "--model",
            "qwen3:8b",
            "--role",
            "context_qa",
            "--repeat",
            "3",
            "--no-cold",
            "--warm",
            "--output",
            "result.jsonl",
        ]
    )

    assert args.command == "benchmark"
    assert args.benchmark_command == "run"
    assert args.repeat == 3
    assert args.cold is False
    assert args.warm is True


def test_benchmark_list_models_does_not_parse_onec_arguments() -> None:
    args = _parser().parse_args(["benchmark", "list-models"])
    assert args.benchmark_command == "list-models"


def test_serve_uses_configured_host_and_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DND_ONEC_BASE_URL", "http://127.0.0.1/assistant/v1")
    monkeypatch.setenv("DND_ONEC_USERNAME", "assistant")
    monkeypatch.setenv("DND_ONEC_PASSWORD", "secret")
    monkeypatch.setenv("DND_AGENT_MODEL", "qwen3:8b")
    monkeypatch.setenv("DND_SERVER_HOST", "127.0.0.1")
    monkeypatch.setenv("DND_SERVER_PORT", "8123")
    captured: dict[str, Any] = {}

    def fake_run(app: object, *, host: str, port: int) -> None:
        captured.update(app=app, host=host, port=port)

    monkeypatch.setattr("dnd_ai_bridge.cli.uvicorn.run", fake_run)

    assert main(["serve"]) == 0
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 8123
    assert captured["app"] is not None


def test_serve_reports_configuration_error_without_starting_uvicorn(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    for name in (
        "DND_ONEC_BASE_URL",
        "DND_ONEC_USERNAME",
        "DND_ONEC_PASSWORD",
        "DND_AGENT_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)

    def unexpected_run(*args: object, **kwargs: object) -> None:
        pytest.fail("uvicorn must not start with invalid configuration")

    monkeypatch.setattr("dnd_ai_bridge.cli.uvicorn.run", unexpected_run)

    assert main(["serve"]) == 2
    assert "Invalid server configuration" in capsys.readouterr().err
