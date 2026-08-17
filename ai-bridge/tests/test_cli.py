from __future__ import annotations

import json

import pytest

from dnd_ai_bridge.cli import _print_json


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
