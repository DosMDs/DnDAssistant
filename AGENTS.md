# D&D Assistant developer context

This file is the repository-wide context for developers and AI collaborators.
Detailed architecture, current implementation status, and contracts are in
[docs/architecture.md](docs/architecture.md).

## Product

D&D Assistant is a completely offline assistant for a player who maintains
multiple D&D campaigns. Its responsibilities include recording and analysing
game events, the campaign journal, entities and their aliases and relations,
game context and time, calendar/scheduling, reference lookup, and analysis of
accumulated data.

D&D 5e (2014) is domain context. Automating D&D mechanics is not a primary
system responsibility and must not drive the architecture.

## Supported environment

- 1C:Enterprise 8.3.27+
- 1C:EDT 2026.1.2
- Python 3.12+
- Guaranteed: Windows 11 on AMD and current macOS on Apple Silicon (target:
  M4 Pro, 24 GB)
- Desirable additional platform: Linux on AMD
- AI inference: local Ollama, completely offline

Do not introduce mandatory cloud services or OS-specific shell/path behaviour.

## Architecture ownership

- **1C** owns UI, business/domain logic, persistent state, campaigns,
  entities, journal, entity relations, aliases, game time/calendar, the source
  of truth for game data, and AI tool semantics and schemas.
- **Python `ai-bridge`** owns:
  - typed HTTP clients;
  - provider adapters;
  - benchmark;
  - cross-platform runtime glue;
  - application-level agent orchestration;
  - planned model routing.
- **Ollama** performs local model inference only.

## Architecture invariants

- 1C is the source of truth for persistent game data.
- Python remains stateless where practical.
- Python does not duplicate 1C domain logic without explicit architectural
  justification.
- AI tool definitions are obtained dynamically from the 1C `/tools` API.
- Do not maintain a parallel hardcoded tool registry in Python.
- A `ModelProvider`/provider executes exactly one completion, streaming or
  non-streaming.
- Tool execution and iteration belong to the application-level agent runtime.
- Actual model identifiers must not be hardcoded in 1C.
- Logical model profiles are `fast` and `large`; their concrete model mapping
  belongs outside 1C and must be benchmark-driven.
- The required runtime is fully offline.
- No mandatory OS-specific shell or path behaviour is allowed.
- Public REST APIs are versioned.
- Performance decisions are based on measured benchmarks.
- Changes to 1C/Python contracts must be evaluated on both sides.
- A D&D rules engine is not a core system responsibility.

## Development responsibilities

- **ChatGPT:** architecture, decomposition, and cross-layer consistency/review.
- **Codex:** Python implementation/tests and repository documentation.
- **1C:Workmate (1С:Напарник):** BSL implementation and refactoring; see
  [.workmate/WORKMATE.md](.workmate/WORKMATE.md).
- **Developer:** metadata changes in EDT, approvals, and runtime/integration
  testing.

## Development environment:
- Python development is performed locally in VS Code.
- Codex is used locally from the VS Code development workflow.
- Codex should operate on the existing local repository checkout.
- Commands and verification steps should be suitable for the integrated/local terminal.

1C metadata objects are created or changed manually by the developer through
EDT. Do not instruct automated agents to manufacture metadata XML directly as
the normal development workflow. If a task requires metadata, identify the
required changes first and wait for them to exist before implementing dependent
BSL.
