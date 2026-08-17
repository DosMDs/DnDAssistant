"""Small diagnostic command-line interface."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import ValidationError

from .config import BridgeSettings
from .errors import BridgeError
from .onec_client import OneCClient


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dnd-ai-bridge")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("health", help="check 1C API health and version")
    subparsers.add_parser("tools", help="list tool descriptors")
    call = subparsers.add_parser("call", help="call a tool")
    call.add_argument("name", help="tool name reported by /tools")
    call.add_argument("arguments", nargs="?", default="{}", help="JSON object")
    return parser


def _print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def _configure_utf8_output() -> None:
    """Make diagnostic JSON UTF-8 even in a legacy Windows console."""

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")


async def _run(args: argparse.Namespace) -> int:
    try:
        settings = BridgeSettings()
    except ValidationError:
        print(
            "Invalid configuration: check DND_ONEC_BASE_URL, DND_ONEC_USERNAME, "
            "DND_ONEC_PASSWORD and DND_ONEC_TIMEOUT_SECONDS.",
            file=sys.stderr,
        )
        return 2

    call_arguments: Mapping[str, Any] | None = None
    if args.command == "call":
        try:
            parsed = json.loads(args.arguments)
        except json.JSONDecodeError as exc:
            print(f"Invalid arguments JSON: {exc.msg}", file=sys.stderr)
            return 2
        if not isinstance(parsed, dict):
            print("Tool arguments must be a JSON object.", file=sys.stderr)
            return 2
        call_arguments = parsed

    try:
        async with OneCClient(settings) as client:
            if args.command == "health":
                result: Any = await client.health()
            elif args.command == "tools":
                result = await client.list_tools()
            else:
                result = await client.call_tool(args.name, call_arguments)
    except BridgeError as exc:
        print(f"Bridge error: {exc}", file=sys.stderr)
        return 1

    if isinstance(result, list):
        output = [item.model_dump(mode="json") for item in result]
    else:
        output = result.model_dump(mode="json")
    _print_json(output)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    _configure_utf8_output()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _parser().parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
