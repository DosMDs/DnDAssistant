# D&D Assistant AI bridge — P04

Асинхронный Python-клиент локального HTTP API 1С с динамическим реестром
инструментов, а также transport/provider layer для локального Ollama native
tool calling. Этап P04 выполняет ровно один model completion и не содержит
agent runtime, выполнения LLM tool calls, повторных запросов к модели,
benchmark scoring или web server.

Архитектура системы и правила разработки описаны в
[../docs/architecture.md](../docs/architecture.md) и
[../AGENTS.md](../AGENTS.md).

## Установка

Требуется Python 3.12 или новее.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
```

Скопируйте `.env.example` в `.env`, замените пример publication и отдельно
добавьте пароль (он намеренно отсутствует в шаблоне):

```dotenv
DND_ONEC_PASSWORD=your-local-password
```

Поддерживаются переменные:

- `DND_ONEC_BASE_URL` — полный базовый путь до `/assistant/v1`;
- `DND_ONEC_USERNAME` — пользователь публикации 1С;
- `DND_ONEC_PASSWORD` — пароль публикации 1С;
- `DND_ONEC_TIMEOUT_SECONDS` — timeout одного HTTP-запроса, по умолчанию 10.
- `DND_OLLAMA_BASE_URL` — локальный Ollama endpoint, по умолчанию
  `http://127.0.0.1:11434`;
- `DND_OLLAMA_TIMEOUT_SECONDS` — timeout Ollama-запроса, по умолчанию 120.

`.env` исключён из Git. Пароль хранится в `SecretStr`, не выводится CLI и не
попадает в логи.

## Использование

```python
from dnd_ai_bridge import (
    BridgeSettings,
    ChatMessage,
    ChatRole,
    ModelRequest,
    OneCClient,
    OllamaClient,
    OllamaProvider,
    OllamaSettings,
    ToolRegistry,
    to_ollama_tools,
)

settings = BridgeSettings()

async with OneCClient(settings) as onec:
    registry = ToolRegistry(onec)
    tools = await registry.load_tools()
    ollama_tools = to_ollama_tools(tools)
    result = await onec.call_tool("get_current_context", {})

async with OllamaClient(OllamaSettings()) as ollama:
    provider = OllamaProvider(ollama, model="qwen3:8b")
    response = await provider.complete(
        ModelRequest(
            messages=[ChatMessage(role=ChatRole.USER, content="Где Торвальд?")]
        )
    )
```

`OllamaProvider.stream()` отдаёт только provider-neutral visible chunks. Поле
Ollama `thinking` не переносится в них; terminal chunk содержит usage и
performance metrics. Диагностические методы transport: `version()`,
`list_models()` и `show_model()`.

Диагностические команды:

```powershell
python -m dnd_ai_bridge.cli health
python -m dnd_ai_bridge.cli tools
python -m dnd_ai_bridge.cli call get_current_context '{}'
python -m dnd_ai_bridge.cli call search_entities '{"query":"Торвальд","types":["npc"],"limit":10}'
```

JSON печатается в UTF-8 с читаемыми русскими символами. INFO-логи содержат
только метод, endpoint/tool, короткий локальный request id и HTTP status, но не
тела запросов, ответы или credentials.

## Тесты

```powershell
python -m pytest
```

Unit-тесты используют `httpx.MockTransport` и не требуют 1С. Integration-тесты
помечены `integration` и автоматически пропускаются, пока не заданы все три
переменные `DND_ONEC_BASE_URL`, `DND_ONEC_USERNAME`, `DND_ONEC_PASSWORD`.
