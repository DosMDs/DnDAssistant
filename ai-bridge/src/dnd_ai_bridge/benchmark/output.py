"""Durable append-only JSONL output for independent benchmark runs."""

from __future__ import annotations

import json
import os
from pathlib import Path

from .models import BenchmarkResult


class JsonlResultWriter:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, result: BenchmarkResult) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(
            result.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
        ) + "\n"
        with self.path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())
