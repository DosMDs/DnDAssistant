from __future__ import annotations

import json

import pytest

from dnd_ai_bridge.cli import _parser, _print_json


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
