# D&D Assistant architecture

Status baseline: **after P07**. This document deliberately distinguishes the
working system (**IMPLEMENTED**) from intended later stages (**PLANNED**).

## Purpose

This is the canonical detailed architecture for D&D Assistant. It defines
component ownership, integration contracts, invariants, supported environments,
and the boundary between the completed P07 local service and planned later
integration layers.

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
- Conversation persistence, Python-side authentication, or write-tool
  execution in the P07 baseline.

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

**IMPLEMENTED after P07:**

```text
1C UI ── POST /v1/agent/run ──> Python FastAPI adapter
                                      |
                                      v
                               AssistantService
                                      |
                                      v
                                AgentRuntime
                                /          \
             1C /assistant/v1 tools       local Ollama inference
```

1C defines what game data means and which AI tools exist. Python loads that
registry for each transient agent run, iterates bounded model completions and
read-only tool calls, and translates between versioned application contracts
and a model provider. The local FastAPI adapter exposes that use case to 1C but
contains no orchestration logic. Ollama only runs inference. The provider still
performs exactly one completion and never executes tools.

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

**IMPLEMENTED after P07:**

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
- transient application-level agent orchestration with dynamic tool loading,
  sequential read-only tool execution, explicit limits, typed results and
  errors, and immediate cancellation propagation;
- a framework-neutral `AssistantService` application boundary;
- a local FastAPI service with process health, versioned agent-run endpoint,
  correlation IDs, stable errors, contextual request logging, and preserved
  cancellation;
- a lifespan composition root that reuses and closes HTTP clients and runtime
  dependencies;
- the `dnd-ai-bridge serve` command with loopback host defaults;
- unit tests and opt-in live 1C integration tests.

**PLANNED, not present after P07:**

- logical-profile routing and concrete model selection policy;
- 1C UI integration;
- user-facing agent streaming and write-tool policy.

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

Python owns transient integration objects and the in-memory transcript for one
agent run. Ollama owns only inference-time model state. The P07 service adds no
conversation persistence; any future persistence needs explicitly designed
ownership and must not silently become a second store of campaign data.

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

## Python local HTTP API

**IMPLEMENTED in P07:** FastAPI exposes the transient assistant use case. The
server defaults to `127.0.0.1:8000`; `DND_SERVER_HOST` and `DND_SERVER_PORT`
are Python-side deployment settings. `DND_AGENT_MODEL` selects one concrete
local model until benchmark-driven `fast`/`large` routing is implemented. 1C
does not send or store that physical identifier.

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Cheap Python process health; no 1C/Ollama probing. |
| `POST` | `/v1/agent/run` | Run `AssistantService` once for a non-empty message list. |

The dependency direction is `HTTP DTO -> AssistantService -> AgentRuntime`.
The endpoint does not create clients or implement the model/tool loop. FastAPI
lifespan constructs one reusable graph of `OneCClient`, `OllamaClient`,
`OllamaProvider`, `ToolRegistry`, `AgentRuntime`, and `AssistantService`, then
closes both transports on shutdown.

Every HTTP request uses `X-Request-ID`: a supplied value is preserved and an
absent value is generated. The response header and agent response/error body
contain the same value. Request logging includes correlation ID, method, path,
status, and duration without logging the prompt or credentials.

Successful agent responses are `{response, request_id}`. User-facing failures
are `{error: {code, message}, request_id}`. Invalid DTOs use
`invalid_request`; unexpected exceptions use a sanitized `internal_error`.
Stable P06 runtime codes are not renamed. Model/policy/empty-output failures
use HTTP 502, configured limit failures use 422, and a 1C tool transport
failure uses 503. Transport status is not the stable application classifier.
Pure ASGI request middleware and the application layers allow
`asyncio.CancelledError` to propagate without an error envelope or retry.

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
decide when an agent loop ends. Those operations belong to the implemented
application-level `AgentRuntime`.

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

At the application boundary, `AgentError` distinguishes unknown tools,
read-only policy rejection, iteration and tool-call limits, an empty terminal
response, and failures while calling a tool through 1C. Normal unsuccessful
`ToolResult` values, including `invalid_arguments`, remain model-visible tool
messages. 1C boundary failures are chained as `tool_transport_failure` without
including arguments or results in the wrapper text. `asyncio` cancellation is
not converted to an agent error and stops further model/tool calls.

At the inbound Python HTTP boundary, all stable `AgentError.code` values are
preserved in the error envelope. Validation and unexpected errors use
`invalid_request` and `internal_error` respectively. Unexpected exception text,
tracebacks, filesystem paths, and credentials are not returned to the client;
the chained exception remains available to server-side logging.

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
streaming, metrics, the CLI, agent orchestration, application service, ASGI
contracts, correlation IDs, error mapping, cancellation, invocation count, and
lifespan cleanup. They use mocked/ASGI HTTP transport or fakes and do not
require 1C or Ollama. Opt-in integration tests exercise a live 1C publication
when all required connection variables are set.

Documentation and contract reviews must compare endpoint names, DTOs, tool
names, and error semantics on both sides. The developer performs 1C runtime and
cross-platform integration testing. P05/P06/P07 tests must keep provider
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

## Agent runtime (implemented P06)

**IMPLEMENTED.** `dnd_ai_bridge.agent.AgentRuntime` sits above
`ModelProvider`, `ToolRegistry`, and `OneCClient`. Each run loads a fresh dynamic
registry, sends provider-neutral requests, preserves assistant tool-call
messages, executes requested tools sequentially, appends deterministic compact
JSON `role=tool` messages, and continues until a non-empty final assistant
message or a structured error.

Only names present in that run's registry can reach `OneCClient`, and P06 only
allows definitions marked `read_only=true`. Argument schema and semantic
validation remain in 1C. `AgentLimits` bounds model completions (iterations) and
total tool calls; there are no retries or parallel tool calls. State and the
returned transcript are transient and in-memory. Cancellation propagates
immediately. User-facing partial-output streaming, write-tool policy, and
persistence are not part of this layer.

## Local service (implemented P07)

**IMPLEMENTED.** `AssistantService` invokes `AgentRuntime` exactly once and
returns only the final visible content to its caller. The FastAPI adapter owns
HTTP DTO validation, request correlation, response envelopes, and status
mapping. Composition and transport cleanup are process-lifecycle concerns, not
per-request work. The service is non-streaming and stateless; conversation
persistence, authentication, UI integration, routing, and write confirmation
remain outside P07.

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
- **P06 — IMPLEMENTED:** application-level agent runtime, per-run dynamic tool
  loading, sequential read-only 1C tool execution, bounded model/tool
  iteration, typed orchestration errors/results, and cancellation semantics.
- **P07 — IMPLEMENTED:** local FastAPI service, framework-neutral application
  service, lifespan composition, request correlation, stable error envelopes,
  cancellation-safe ASGI handling, and the `serve` CLI command.
- **PLANNED after P07:** 1C UI integration, benchmark-driven model-profile
  routing, user-facing agent streaming, and a separately designed write-tool
  policy.
