from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from dnd_ai_bridge.config import BridgeSettings
from dnd_ai_bridge.errors import (
    OneCAuthenticationError,
    OneCProtocolError,
    OneCTransportError,
)
from dnd_ai_bridge.onec_client import OneCClient


def settings() -> BridgeSettings:
    return BridgeSettings(
        onec_base_url="http://127.0.0.1/demo/hs/assistant/v1/",
        onec_username="assistant",
        onec_password="top-secret",
        onec_timeout_seconds=1,
    )


def client_for(handler: Any) -> OneCClient:
    return OneCClient(settings(), transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_health_accepts_expected_contract() -> None:
    async with client_for(
        lambda request: httpx.Response(
            200, json={"status": "ok", "api_version": "1"}, request=request
        )
    ) as client:
        response = await client.health()

    assert response.status == "ok"
    assert response.api_version == "1"


@pytest.mark.asyncio
async def test_base_url_keeps_v1_path_without_double_slash() -> None:
    seen_path = ""

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal seen_path
        seen_path = request.url.path
        return httpx.Response(
            200, json={"status": "ok", "api_version": "1"}, request=request
        )

    async with client_for(handler) as client:
        await client.health()

    assert seen_path == "/demo/hs/assistant/v1/health"


@pytest.mark.asyncio
async def test_health_rejects_unknown_api_version() -> None:
    async with client_for(
        lambda request: httpx.Response(
            200, json={"status": "ok", "api_version": "2"}, request=request
        )
    ) as client:
        with pytest.raises(OneCProtocolError, match="Unsupported"):
            await client.health()


@pytest.mark.asyncio
async def test_health_rejects_non_ok_status() -> None:
    async with client_for(
        lambda request: httpx.Response(
            200, json={"status": "degraded", "api_version": "1"}, request=request
        )
    ) as client:
        with pytest.raises(OneCProtocolError, match="Unexpected /health status"):
            await client.health()


@pytest.mark.asyncio
async def test_health_rejects_malformed_json() -> None:
    async with client_for(
        lambda request: httpx.Response(200, text="not-json", request=request)
    ) as client:
        with pytest.raises(OneCProtocolError, match="not valid JSON"):
            await client.health()


@pytest.mark.asyncio
async def test_health_maps_401_to_authentication_error() -> None:
    async with client_for(
        lambda request: httpx.Response(401, text="login page", request=request)
    ) as client:
        with pytest.raises(OneCAuthenticationError) as error:
            await client.health()

    assert error.value.http_status == 401
    assert "login page" not in str(error.value)
    assert "top-secret" not in str(error.value)


@pytest.mark.asyncio
async def test_403_is_also_a_distinct_authentication_error() -> None:
    async with client_for(
        lambda request: httpx.Response(403, text="forbidden", request=request)
    ) as client:
        with pytest.raises(OneCAuthenticationError) as error:
            await client.health()

    assert error.value.code == "authorization_failed"
    assert error.value.http_status == 403


@pytest.mark.asyncio
async def test_health_maps_structured_500_to_transport_error() -> None:
    body = {
        "success": False,
        "data": None,
        "error": {"code": "internal_error", "message": "Сбой 1С"},
    }
    async with client_for(
        lambda request: httpx.Response(500, json=body, request=request)
    ) as client:
        with pytest.raises(OneCTransportError) as error:
            await client.health()

    assert error.value.code == "internal_error"
    assert error.value.http_status == 500
    assert str(error.value) == "Сбой 1С"


@pytest.mark.asyncio
async def test_invalid_html_error_body_is_not_copied_to_exception() -> None:
    body = "<html>very sensitive publication error page</html>"
    async with client_for(
        lambda request: httpx.Response(500, text=body, request=request)
    ) as client:
        with pytest.raises(OneCTransportError) as error:
            await client.health()

    assert error.value.code == "invalid_transport_error"
    assert "sensitive" not in str(error.value)
    assert error.value.http_status == 500


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 413, 415])
async def test_expected_client_http_errors_are_transport_errors(status: int) -> None:
    body = {
        "success": False,
        "data": None,
        "error": {"code": "bad_request", "message": "Некорректный запрос"},
    }
    async with client_for(
        lambda request: httpx.Response(status, json=body, request=request)
    ) as client:
        with pytest.raises(OneCTransportError) as error:
            await client.call_tool("anything", {})

    assert error.value.http_status == status


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("raised", "expected_code"),
    [
        (httpx.ConnectTimeout("too slow"), "timeout"),
        (httpx.ConnectError("offline"), "connection_error"),
    ],
)
async def test_network_failures_are_safe_transport_errors(
    raised: httpx.RequestError, expected_code: str
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raised.request = request
        raise raised

    async with client_for(handler) as client:
        with pytest.raises(OneCTransportError) as error:
            await client.health()

    assert error.value.code == expected_code
    assert "top-secret" not in str(error.value)


@pytest.mark.asyncio
async def test_list_tools_parses_arbitrary_descriptors_and_preserves_schema() -> None:
    descriptors = [
        {
            "name": f"tool_{index}",
            "description": f"Инструмент {index}",
            "read_only": index % 2 == 0,
            "input_schema": {
                "type": "object",
                "properties": {"поле": {"const": index}},
                "x-custom": [index, "значение"],
            },
        }
        for index in range(5)
    ]
    body = {"success": True, "data": {"tools": descriptors}, "error": None}
    async with client_for(
        lambda request: httpx.Response(200, json=body, request=request)
    ) as client:
        tools = await client.list_tools()

    assert len(tools) == 5
    assert tools[3].input_schema == descriptors[3]["input_schema"]
    assert tools[0].read_only is True
    assert tools[1].read_only is False


@pytest.mark.asyncio
async def test_list_tools_treats_success_false_as_protocol_failure() -> None:
    body = {
        "success": False,
        "data": None,
        "error": {"code": "internal", "message": "failure"},
    }
    async with client_for(
        lambda request: httpx.Response(200, json=body, request=request)
    ) as client:
        with pytest.raises(OneCProtocolError, match="success=false"):
            await client.list_tools()


@pytest.mark.asyncio
async def test_list_tools_requires_data_tools() -> None:
    body = {"success": True, "data": {}, "error": None}
    async with client_for(
        lambda request: httpx.Response(200, json=body, request=request)
    ) as client:
        with pytest.raises(OneCProtocolError, match="Invalid /tools response"):
            await client.list_tools()


@pytest.mark.asyncio
async def test_list_tools_requires_data_envelope() -> None:
    body = {"success": True, "data": None, "error": None}
    async with client_for(
        lambda request: httpx.Response(200, json=body, request=request)
    ) as client:
        with pytest.raises(OneCProtocolError, match="data.tools"):
            await client.list_tools()


@pytest.mark.asyncio
async def test_list_tools_rejects_invalid_json() -> None:
    async with client_for(
        lambda request: httpx.Response(200, content=b"{", request=request)
    ) as client:
        with pytest.raises(OneCProtocolError, match="not valid JSON"):
            await client.list_tools()


@pytest.mark.asyncio
async def test_call_tool_returns_success_result() -> None:
    body = {"success": True, "data": {"x": 1}, "error": None}
    async with client_for(
        lambda request: httpx.Response(200, json=body, request=request)
    ) as client:
        result = await client.call_tool("any_tool", {"not": "validated"})

    assert result.success is True
    assert result.data == {"x": 1}


@pytest.mark.asyncio
@pytest.mark.parametrize("code", ["invalid_arguments", "unknown_tool"])
async def test_tool_error_is_returned_without_exception(code: str) -> None:
    body = {
        "success": False,
        "data": None,
        "error": {"code": code, "message": "Ошибка инструмента"},
    }
    async with client_for(
        lambda request: httpx.Response(200, json=body, request=request)
    ) as client:
        result = await client.call_tool("abracadabra", {"foo": 1})

    assert result.success is False
    assert result.error is not None
    assert result.error.code == code


@pytest.mark.asyncio
async def test_call_tool_rejects_invalid_server_envelope() -> None:
    body = {
        "success": True,
        "data": {"x": 1},
        "error": {"code": "impossible", "message": "broken"},
    }
    async with client_for(
        lambda request: httpx.Response(200, json=body, request=request)
    ) as client:
        with pytest.raises(OneCProtocolError, match="Invalid tool result"):
            await client.call_tool("broken", {})


@pytest.mark.asyncio
async def test_none_arguments_are_sent_as_empty_object() -> None:
    received: Any = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal received
        received = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={"success": True, "data": None, "error": None},
            request=request,
        )

    async with client_for(handler) as client:
        await client.call_tool("get_current_context")

    assert received == {}


@pytest.mark.asyncio
async def test_unicode_request_and_response_round_trip() -> None:
    phrases = [
        "Торвальд Железнорукий",
        "5 Миртула 1492 ЛД",
        "Гильдия кузнецов",
    ]
    received: Any = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal received
        received = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={"success": True, "data": {"values": phrases}, "error": None},
            request=request,
        )

    async with client_for(handler) as client:
        result = await client.call_tool("поиск", {"query": phrases[0]})

    assert received == {"query": phrases[0]}
    assert result.data == {"values": phrases}


@pytest.mark.asyncio
async def test_request_uses_basic_auth_without_logging_credentials(
    caplog: pytest.LogCaptureFixture,
) -> None:
    authorization = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal authorization
        authorization = request.headers.get("Authorization")
        return httpx.Response(
            200, json={"status": "ok", "api_version": "1"}, request=request
        )

    async with client_for(handler) as client:
        await client.health()

    assert authorization is not None
    assert authorization.startswith("Basic ")
    assert "top-secret" not in caplog.text
