# D&D Assistant architecture

Status baseline: **after P05**. This document deliberately distinguishes the
working system (**IMPLEMENTED**) from intended later stages (**PLANNED**).

## Purpose

This is the canonical detailed architecture for D&D Assistant. It defines
component ownership, integration contracts, invariants, supported environments,
and the boundary between the completed P05 benchmark work and the planned P06
agent layer.

Repository-wide working rules are summarised in [AGENTS.md](../AGENTS.md).

## Product scope

D&D Assistant is a completely offline player assistant for maintaining multiple
D&D campaigns. Product scope includes:

- recording and analysing game events;
- a campaign journal;
- campaign entities, aliases, and relations;
- current game context and game time;
- calendar and scheduling;
- reference information;
- analysis of accumulated campaign data.

D&D 5e (2014) supplies vocabulary and subject context. The architecture is
centred on campaign knowledge and assistant workflows, not rules automation.

## Non-goals

- A comprehensive D&D rules engine or primary automation of game mechanics.
- Cloud inference, a mandatory internet connection, or a cloud fallback.
- Moving persistent campaign state or domain authority out of 1C.
- Reimplementing 1C business rules in Python for convenience.
- Treating Ollama as an application or persistence layer.
- A bridge web server or agent loop in the P05 baseline.

## Supported environments

- 1C:Enterprise 8.3.27 or newer.
- 1C:EDT 2026.1.2.
- Python 3.12 or newer (`ai-bridge/pyproject.toml` enforces `>=3.12`).
- Guaranteed operating systems:
  - Windows 11 on AMD;
  - current macOS on Apple Silicon, with M4 Pro and 24 GB as the target.
- Desirable additional operating system: Linux on AMD.
- AI runtime: a local Ollama installation with locally available models.

## High-level architecture

**IMPLEMENTED after P05:**

```text
1C application and persistent data
        |
        | local, versioned /assistant/v1 HTTP API
        v
Python ai-bridge: typed 1C client, dynamic tools, neutral model boundary
        |
        | local Ollama HTTP API
        v
Ollama: one local model completion
```

1C defines what game data means and which AI tools exist. Python translates
between versioned application contracts and a model provider. Ollama only runs
inference. The current provider returns model output or tool-call requests; no
component in P05 iterates those calls into an agent conversation.

**PLANNED:** the P06 application-level agent runtime will sit in Python above
`ModelProvider`, `OneCClient`, and `ToolRegistry`. It will not move provider or
domain responsibilities across their existing boundaries.

## 1C responsibilities

**IMPLEMENTED ownership:**

- UI and user interaction;
- business and domain logic;
- persistent state and the source of truth for game data;
- campaigns and active-campaign isolation;
- journal and game events;
- entities, aliases, and entity relations;
- game context, game time, calendar, and scheduling;
- AI-facing tool names, descriptions, argument JSON Schemas, stable DTO/type
  codes, validation, dispatch, and result semantics;
- the versioned Assistant HTTP API.

The AI-facing common modules call existing domain modules such as campaign
management, assistant context, entity search, entity relations, and the game
calendar. The HTTP adapter handles transport concerns and does not contain
entity or campaign logic.

## Python `ai-bridge` responsibilities

**IMPLEMENTED after P05:**

- environment-backed connection settings;
- an asynchronous typed `OneCClient` for health, tool discovery, and tool
  calls;
- a dynamic `ToolRegistry` that validates and neutrally maps descriptors
  received from 1C;
- provider-neutral request, response, message, tool-call, usage, and
  performance DTOs;
- the `ModelProvider` protocol;
- `OllamaClient`, `OllamaProvider`, and mapping to Ollama native tool calling;
- streaming of visible model output;
- diagnostic CLI commands for the 1C API;
- transport methods for Ollama version, model list, and model details;
- typed running-model discovery through Ollama `GET /api/ps`;
- provider-specific generation settings outside neutral `ModelRequest`;
- versioned synthetic scenarios, deterministic scoring, verified cold/warm
  orchestration, environment/model metadata, and durable JSONL output;
- unit tests and opt-in live 1C integration tests.

**PLANNED, not present after P05:**

- logical-profile routing and concrete model selection policy;
- application-level agent orchestration and tool iteration (P06).

Python should remain stateless where practical. Persistent game state and
domain decisions must not be copied into the bridge without an explicit
architectural justification.

## Ollama responsibilities

**IMPLEMENTED integration:** Ollama receives a typed chat request through its
local HTTP API and performs exactly one non-streaming or streaming completion.
It may return visible content and provider-native tool calls, which the provider
maps into neutral DTOs.

Ollama does not own campaigns, tool semantics, tool execution, conversation
iteration, model-routing policy, or persistent application state. The current
client has no authentication, retries, proxies, or cloud fallback and sets
`trust_env=False` for its HTTP transport.

## Data ownership

1C is the only source of truth for persistent campaign and game data. Calls that
depend on a campaign resolve the active campaign in 1C and constrain their
domain queries accordingly. Tool results are snapshots for an individual call;
they are not a Python-side database.

Python owns transient integration objects and provider-neutral messages for a
completion. Ollama owns only inference-time model state. Any future agent
conversation state must remain transient or have explicitly designed ownership;
it must not silently become a second store of campaign data.

## 1C Assistant HTTP API

**IMPLEMENTED:** the HTTP service root is `/assistant`; the current public API
version is `v1`.

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/assistant/v1/health` | Returns transport health and `api_version: "1"`. |
| `GET` | `/assistant/v1/tools` | Returns the dynamic tool registry from 1C. |
| `POST` | `/assistant/v1/tools/{name}` | Executes one explicitly allowed tool with a JSON-object body. |

The Python base URL points through `/assistant/v1`; `OneCClient` appends
`health`, `tools`, or `tools/{name}`. The client uses the configured 1C
publication credentials.

Successful registry and tool-call responses use a `{success, data, error}`
envelope. Tool-level failures also use this envelope with HTTP 200. The health
response is the compact `{status, api_version}` object.

The HTTP adapter owns JSON parsing, content type, request-size limits,
serialization, and transport failures. The 1C tool layer owns argument
validation and tool semantics. Contract changes require coordinated review of
the 1C producer and Python consumer.

## AI tool registry

**IMPLEMENTED:** `GET /assistant/v1/tools` obtains its descriptors directly
from `ИнструментыАссистентаСервер`. Each descriptor contains a stable name,
description, `read_only`, and `input_schema`. Python loads these dynamically and
preserves the JSON Schema while mapping it into provider-neutral and then
Ollama-native definitions.

The current explicit 1C whitelist contains five read-only tools:

- `get_current_context`;
- `search_entities`;
- `get_entity`;
- `get_relations`;
- `get_calendar_agenda`.

1C dispatches these names through explicit branches. Arbitrary BSL invocation
from an AI-supplied name is forbidden. Python must not introduce a second
hardcoded registry: 1C remains authoritative for names, schemas, validation,
stable type codes, campaign isolation, and execution semantics.

## Provider-neutral model boundary

**IMPLEMENTED:** `ModelProvider` exposes:

- `complete(ModelRequest) -> ModelResponse` for one completion;
- `stream(ModelRequest) -> AsyncIterator[ModelStreamChunk]` for one streaming
  completion.

Neutral messages support system, user, assistant, and tool roles. Neutral tool
definitions and model-requested tool calls do not expose Ollama DTOs to the
application layer.

A provider performs exactly one model completion. It may return requested tool
calls, but it does not execute them, append their results, retry the model, or
decide when an agent loop ends. Those operations belong to the planned P06
agent runtime.

## Ollama transport/provider

**IMPLEMENTED after P05:**

- `OllamaClient.chat()` calls `POST /api/chat` for a non-streaming completion;
- `OllamaClient.stream_chat()` reads and validates the NDJSON stream from
  `POST /api/chat` and requires a final chunk;
- `OllamaClient.version()` calls `GET /api/version`;
- `OllamaClient.list_models()` calls `GET /api/tags`;
- `OllamaClient.show_model()` calls `POST /api/show`;
- `OllamaClient.running_models()` calls `GET /api/ps`;
- benchmark unloads use the HTTP API with `keep_alive=0`, never `ollama ps`;
- `OllamaProvider` maps neutral messages, schemas, and tool calls without
  executing tools;
- `OllamaGenerationSettings` maps temperature, seed, context allocation, and
  keep-alive without changing the provider-neutral request;
- thinking-only chunks are not exposed as visible neutral output;
- terminal output carries usage and client-observed performance metrics.

The provider currently receives one concrete model identifier in its
constructor. Logical `fast`/`large` profile resolution is not yet implemented.

## Offline requirements

The required runtime is completely offline:

- 1C, Python, Ollama, and model files must be usable without internet access;
- no cloud API or telemetry service may be mandatory;
- there must be no automatic cloud fallback;
- production and benchmark Ollama processes must set `OLLAMA_NO_CLOUD=1`, and
  benchmark input rejects explicitly cloud model identifiers;
- normal inference and tool calls remain within the local machine or an
  explicitly local/offline deployment boundary;
- dependency and model acquisition may be a separate setup activity, but is not
  part of runtime operation.

The existing Ollama client is local by default (`127.0.0.1`) and ignores proxy
environment variables. The 1C client accepts an absolute HTTP(S) publication
URL and warns when its host is not loopback; deployment policy must still
preserve the offline boundary.

## Cross-platform requirements

Runtime code and documentation must not require a particular shell, drive
letter, path separator, or architecture-specific binary behaviour. Platform
integration belongs in the Python runtime-glue layer, behind portable Python
interfaces where possible.

Changes must be evaluated for the two guaranteed targets: Windows 11 AMD and
current macOS Apple Silicon. Linux AMD compatibility is desirable. Examples may
show a platform-specific activation command, but product runtime behaviour must
not depend on it.

## Error model

**IMPLEMENTED:** transport-level and tool-level failures are separate.

On the 1C boundary:

- malformed JSON, unsupported media type, excessive request size, invalid
  transport requests, and adapter failures use non-200 HTTP statuses and a
  machine-readable error envelope;
- a known or unknown tool that reached the dispatcher returns HTTP 200 with
  `{success: false, data: null, error: {code, message}}` for tool-level errors;
- current stable tool error codes include `unknown_tool`, `invalid_arguments`,
  `no_active_campaign`, `entity_not_found`, `game_time_not_set`, and
  `internal_error`.

`OneCClient` distinguishes authentication/authorization, transport, and
HTTP-200 protocol-contract failures. An unsuccessful valid `ToolResult` remains
a result rather than becoming a transport exception.

On the Ollama boundary, the Python client distinguishes timeout, connection,
non-success HTTP status, and malformed/protocol-invalid responses. Error text is
bounded before it is exposed. Providers do not reinterpret these failures as
tool errors.

## Performance metrics

**IMPLEMENTED measurement primitives:** each completion can report Ollama
server metrics for total/load duration, prompt token evaluation count/duration,
and generated token evaluation count/duration. The provider also records
monotonic client timestamps and derives wall duration and time to the first
meaningful visible content or tool call.

For a non-streaming response, the first meaningful output can only be observed
at completion. For a stream, thinking-only and empty chunks do not count as the
first meaningful visible chunk.

**IMPLEMENTED in P05:** measured runs always stream; JSONL preserves client and
server timings plus safe prompt/generation token rates. Synthetic versioned
cases cover tool selection, tool arguments, context QA, and campaign summaries.
Scoring is deterministic and local. Cold/warm preconditions are confirmed with
`/api/ps`; unconfirmed state makes a run invalid. Model/profile recommendations
still require real measurements on target hardware.

## Testing strategy

**IMPLEMENTED:** Python unit tests cover settings, typed DTO invariants, the 1C
client, dynamic registry, Ollama mapping, transport/provider behaviour,
streaming, metrics, and the diagnostic CLI. They use mocked HTTP transport and
do not require 1C or Ollama. Opt-in integration tests exercise a live 1C
publication when all required connection variables are set.

Documentation and contract reviews must compare endpoint names, DTOs, tool
names, and error semantics on both sides. The developer performs 1C runtime and
cross-platform integration testing. P05/P06 tests must keep provider
tests focused on one completion and test orchestration separately.

## Model profiles: `fast` / `large`

The canonical logical profiles are:

- `fast`: prioritises low latency and resource efficiency for routine assistant
  turns and tool-selection steps;
- `large`: prioritises answer quality and more demanding analysis within local
  hardware constraints.

**PLANNED:** profile-to-Ollama-model mapping and routing. Concrete model
identifiers must not be stored or hardcoded in 1C. They belong in Python-side
runtime configuration and must be selected from P05 benchmark evidence for the
target hardware. After P05, callers still pass a concrete model identifier
directly to `OllamaProvider`.

## Benchmark architecture (implemented P05)

**IMPLEMENTED.** The Python-side `dnd_ai_bridge.benchmark` package sits above
the one-completion provider boundary. It runs reproducible synthetic cases
against candidate local models, retains each repetition independently, derives
comparable metrics, and emits append-only versioned JSONL evidence.

Benchmark orchestration and scoring are not part of `ModelProvider`, 1C, or
Ollama. Records identify environment, model metadata, generation configuration,
scenario, mode, repetition, raw/derived metrics, scoring, and expected errors.
The runner never uses 1C data or credentials.

## Agent runtime (planned P06)

**PLANNED; no agent runtime exists after P05.** P06 will add an
application-level Python orchestration layer. Its responsibilities will include
loading current tools from 1C, calling a selected provider, validating requested
tools against the current registry, executing them through `OneCClient`, adding
neutral tool results to the conversation, iterating with explicit limits, and
returning the final outcome.

The runtime must preserve campaign isolation and 1C authority, must not call
arbitrary BSL, and must not turn the provider into an agent. Iteration limits,
cancellation, and orchestration errors belong at this application layer.

## Architecture invariants

1. 1C is the source of truth for persistent game data.
2. Python remains stateless where practical.
3. Python does not duplicate 1C domain logic without explicit architectural
   justification.
4. AI tool definitions are obtained dynamically from the 1C `/tools` API; no
   parallel hardcoded Python registry is allowed.
5. A `ModelProvider`/provider executes exactly one completion.
6. Tool execution and iteration belong to the application-level agent runtime.
7. Actual model identifiers are not hardcoded in 1C.
8. Logical model profiles are `fast` and `large`.
9. Required runtime operation is fully offline.
10. No mandatory OS-specific shell/path behaviour is allowed.
11. Public REST APIs are versioned.
12. Performance decisions are based on measured benchmarks.
13. Contract changes are evaluated on both the 1C and Python sides.
14. A D&D rules engine is not a core system responsibility.

## Development workflow

- ChatGPT owns architecture, decomposition, and cross-layer consistency/review.
- Codex owns Python implementation/tests and repository documentation.
- 1C:Workmate owns BSL implementation and refactoring under the specialised
  rules in [.workmate/WORKMATE.md](../.workmate/WORKMATE.md).
- The developer owns approvals, manual 1C metadata changes in EDT, and
  runtime/integration testing.

1C metadata objects and attributes are created or changed manually through EDT.
Automated agents must not manufacture metadata XML as the normal workflow. When
a change needs metadata, first enumerate the required objects/attributes; BSL
that depends on them follows after the developer has added them.

Every HTTP/tool contract change must state its impact on both layers and update
tests and this document in the same change. Generated or service-owned regions
should be left untouched unless the task specifically requires them.

## Current roadmap

- **P04 — IMPLEMENTED:** typed 1C integration, dynamic tool registry,
  provider-neutral model boundary, Ollama transport/provider, native tool-call
  mapping, streaming, usage/performance metrics, and Ollama diagnostic
  endpoints.
- **P04.5 — DOCUMENTATION BASELINE:** canonical developer context and
  cross-layer architecture documentation.
- **P05 — IMPLEMENTED:** offline benchmark runner, deterministic scoring,
  measured streaming metrics, environment/model metadata, verified cold/warm
  state, repetitions, and durable JSONL output.
- **P06 — PLANNED:** application-level agent runtime, 1C tool execution, and
  bounded model/tool iteration.
