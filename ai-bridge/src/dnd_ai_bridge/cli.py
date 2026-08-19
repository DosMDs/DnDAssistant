"""Small diagnostic command-line interface."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from collections.abc import Mapping, Sequence
from functools import partial
from typing import Any

import uvicorn
from pydantic import ValidationError

from .api import create_app
from .benchmark.models import ScenarioRole
from .benchmark.output import JsonlResultWriter
from .benchmark.runner import BenchmarkRunner
from .benchmark.scenarios import load_builtin_scenarios
from .composition import application_resources_lifespan
from .config import AgentSettings, BridgeSettings, OllamaSettings, ServerSettings
from .errors import BridgeError
from .onec_client import OneCClient
from .ollama_client import OllamaClient
from .ollama_provider import OllamaGenerationSettings


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dnd-ai-bridge")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("health", help="check 1C API health and version")
    subparsers.add_parser("tools", help="list tool descriptors")
    subparsers.add_parser("serve", help="run the local assistant HTTP API")
    call = subparsers.add_parser("call", help="call a tool")
    call.add_argument("name", help="tool name reported by /tools")
    call.add_argument("arguments", nargs="?", default="{}", help="JSON object")
    benchmark = subparsers.add_parser(
        "benchmark", help="run reproducible offline Ollama benchmarks"
    )
    benchmark_commands = benchmark.add_subparsers(
        dest="benchmark_command", required=True
    )
    benchmark_commands.add_parser(
        "list-models", help="list locally installed Ollama models"
    )
    run = benchmark_commands.add_parser("run", help="run benchmark scenarios")
    run.add_argument("--model", required=True)
    selection = run.add_mutually_exclusive_group(required=True)
    selection.add_argument("--role", choices=[role.value for role in ScenarioRole])
    selection.add_argument("--all", action="store_true", dest="all_roles")
    run.add_argument("--repeat", type=int, default=1)
    run.add_argument(
        "--cold", action=argparse.BooleanOptionalAction, default=True
    )
    run.add_argument(
        "--warm", action=argparse.BooleanOptionalAction, default=True
    )
    run.add_argument("--output", required=True)
    run.add_argument("--temperature", type=float, default=0.0)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--num-ctx", type=int)
    run.add_argument("--keep-alive")
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
    if args.command == "benchmark":
        return await _run_benchmark(args)

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


def _run_server() -> int:
    """Load all service settings before handing control to Uvicorn."""

    try:
        bridge_settings = BridgeSettings()
        ollama_settings = OllamaSettings()
        agent_settings = AgentSettings()
        server_settings = ServerSettings()
    except ValidationError:
        print(
            "Invalid server configuration: check DND_ONEC_*, DND_OLLAMA_*, "
            "DND_AGENT_MODEL and DND_SERVER_*.",
            file=sys.stderr,
        )
        return 2

    resource_lifespan = partial(
        application_resources_lifespan,
        bridge_settings=bridge_settings,
        ollama_settings=ollama_settings,
        agent_settings=agent_settings,
    )
    app = create_app(resource_lifespan=resource_lifespan)
    uvicorn.run(app, host=server_settings.host, port=server_settings.port)
    return 0


def _keep_alive(value: str | None) -> str | int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return value


async def _run_benchmark(args: argparse.Namespace) -> int:
    """Benchmark configuration is intentionally independent from 1C settings."""

    try:
        settings = OllamaSettings()
    except ValidationError:
        print(
            "Invalid Ollama configuration: check DND_OLLAMA_BASE_URL and "
            "DND_OLLAMA_TIMEOUT_SECONDS.",
            file=sys.stderr,
        )
        return 2

    try:
        async with OllamaClient(settings) as client:
            if args.benchmark_command == "list-models":
                response = await client.list_models()
                _print_json(
                    [model.model_dump(mode="json") for model in response.models]
                )
                return 0

            if args.repeat < 1:
                print("--repeat must be at least 1.", file=sys.stderr)
                return 2
            if not args.cold and not args.warm:
                print("At least one of --cold or --warm is required.", file=sys.stderr)
                return 2
            role = None if args.all_roles else ScenarioRole(args.role)
            scenarios = load_builtin_scenarios(role)
            generation = OllamaGenerationSettings(
                temperature=args.temperature,
                seed=args.seed,
                num_ctx=args.num_ctx,
                keep_alive=_keep_alive(args.keep_alive),
            )
            runner = BenchmarkRunner(client, JsonlResultWriter(args.output))
            results = await runner.run(
                model=args.model,
                scenarios=scenarios,
                repeat=args.repeat,
                cold=args.cold,
                warm=args.warm,
                generation_settings=generation,
            )
    except (BridgeError, ValidationError, ValueError) as exc:
        print(f"Benchmark error: {exc}", file=sys.stderr)
        return 1

    _print_json(
        {
            "output": args.output,
            "records": len(results),
            "errors": sum(result.error is not None for result in results),
        }
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    _configure_utf8_output()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _parser().parse_args(argv)
    if args.command == "serve":
        return _run_server()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
