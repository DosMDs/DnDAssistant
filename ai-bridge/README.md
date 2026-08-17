# D&D Assistant AI bridge — A01

Независимый асинхронный Python-клиент локального HTTP API 1С. Этап A01 не
содержит LLM, model adapters, agent runtime, повторных запросов или web server.

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

`.env` исключён из Git. Пароль хранится в `SecretStr`, не выводится CLI и не
попадает в логи.

## Использование

```python
from dnd_ai_bridge import BridgeSettings, OneCClient

settings = BridgeSettings()

async with OneCClient(settings) as onec:
    tools = await onec.list_tools()
    result = await onec.call_tool("get_current_context", {})
```

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

