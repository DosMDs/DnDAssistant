# D&D Assistant AI bridge — P05

Асинхронный Python-клиент локального HTTP API 1С с динамическим реестром
инструментов, а также transport/provider layer для локального Ollama native
tool calling. Этап P05 также содержит воспроизводимый offline benchmark
локальных моделей. Provider по-прежнему выполняет ровно один model completion;
agent runtime, выполнение LLM tool calls и web server не реализованы.

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

Для production и benchmark Ollama обязательно должен быть запущен с запретом
cloud-функций:

```text
OLLAMA_NO_CLOUD=1
```

Это переменная окружения процесса Ollama, а не настройка bridge. Python не
запускает Ollama и не меняет его системную конфигурацию. Benchmark дополнительно
отклоняет явно cloud model identifiers и не использует cloud API или proxy из
окружения.

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
`list_models()`, `show_model()` и typed `running_models()` (`GET /api/ps`).
Provider-specific `OllamaGenerationSettings` поддерживает `temperature`,
`seed`, `num_ctx` и `keep_alive`, не добавляя Ollama-поля в `ModelRequest`.

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

## Offline benchmark

Benchmark использует встроенные синтетические versioned JSON-сценарии четырёх
ролей: `tool_selection`, `tool_arguments`, `context_qa` и
`campaign_summary`. Versioned fixture tool schemas не загружаются из 1С.
Scoring полностью локальный и детерминированный: точный tool selection,
структурное сравнение аргументов (включая subset mode), required/forbidden
facts после Unicode-нормализации. Другая LLM как judge не используется.

```powershell
python -m dnd_ai_bridge.cli benchmark list-models
python -m dnd_ai_bridge.cli benchmark run --model qwen3:8b --all --repeat 3 --cold --warm --output results/qwen3-8b.jsonl
python -m dnd_ai_bridge.cli benchmark run --model qwen3:8b --role context_qa --no-cold --warm --output results/context.jsonl
```

Benchmark-команды используют только `OllamaSettings` и не требуют credentials
1С. Дополнительные воспроизводимые параметры CLI: `--temperature`, `--seed`,
`--num-ctx`, `--keep-alive` (defaults для benchmark: temperature/seed равны 0).

Cold/warm — проверенные состояния, а не метки намерения:

- перед каждым cold repetition runner отправляет локальный `keep_alive=0`
  unload и требует отсутствия модели в `/api/ps`;
- перед каждым warm repetition выполняется отдельный unmeasured streaming
  warm-up, после которого `/api/ps` обязан подтвердить загруженную модель;
- если `/api/ps` недоступен или состояние не совпало, repetition записывается
  как `invalid_state` и measured completion не начинается;
- warm-up не попадает в measured results; каждый repetition — отдельная
  JSONL-запись.

JSONL — append-only primary artifact. После каждой строки выполняются flush и
`fsync`, поэтому ошибка следующего repetition не уничтожает предыдущие записи.
Пример одной сокращённой строки (реальные duration зависят от системы):

```json
{"schema_version":"1","run_id":"8d12...","scenario_id":"context.synthetic_harbor","scenario_version":1,"role":"context_qa","model":"qwen3:8b","mode":"warm","repeat_index":1,"generation_settings":{"temperature":0.0,"seed":0},"environment":{"os":"Windows","os_version":"10.0.26100","architecture":"AMD64","python_version":"3.12.0","ollama_version":"0.12.6"},"model_info":{"requested_name":"qwen3:8b","reported_name":"qwen3:8b","digest":"sha256:...","parameter_size":"8.2B","quantization":"Q4_K_M","capabilities":["completion","tools"],"context_metadata":{"qwen3.context_length":40960},"allocated_context_length":8192,"size_vram":5550000000},"raw_metrics":{"client_wall_duration_ns":2100000000,"time_to_first_meaningful_chunk_ns":430000000,"total_duration_ns":2050000000,"load_duration_ns":0,"prompt_eval_count":51,"prompt_eval_duration_ns":180000000,"eval_count":24,"eval_duration_ns":1500000000},"derived_metrics":{"prompt_tokens_per_second":283.33,"generation_tokens_per_second":16.0},"scoring":{"passed":true,"checks":{"required:Glass Harbor":true,"required:17 Frostwane":true,"forbidden:Moonport":true,"forbidden:18 Frostwane":true},"message":null},"error":null}
```

Оставшиеся задачи P06: application-level agent runtime, динамическая загрузка
1C tools, проверка и выполнение запрошенных tool calls через `OneCClient`,
добавление tool results в диалог, ограниченный цикл model/tool, cancellation и
ошибки orchestration. Эти обязанности не входят в provider и benchmark.

## Тесты

```powershell
python -m pytest
```

Unit-тесты используют mocks/fakes и не требуют 1С, Ollama или модели.
Integration-тесты
помечены `integration` и автоматически пропускаются, пока не заданы все три
переменные `DND_ONEC_BASE_URL`, `DND_ONEC_USERNAME`, `DND_ONEC_PASSWORD`.
