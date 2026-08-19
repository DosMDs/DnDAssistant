from __future__ import annotations

import pytest

from dnd_ai_bridge.composition import ApplicationResources


class ClosingClient:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.closed = False
        self.error = error

    async def close(self) -> None:
        self.closed = True
        if self.error is not None:
            raise self.error


async def test_application_resources_close_both_reusable_clients() -> None:
    onec = ClosingClient()
    ollama = ClosingClient()
    resources = ApplicationResources(
        assistant_service=object(),  # type: ignore[arg-type]
        onec_client=onec,  # type: ignore[arg-type]
        ollama_client=ollama,  # type: ignore[arg-type]
    )

    await resources.close()

    assert onec.closed is True
    assert ollama.closed is True


async def test_application_resources_close_onec_if_ollama_close_fails() -> None:
    onec = ClosingClient()
    ollama = ClosingClient(error=RuntimeError("close failed"))
    resources = ApplicationResources(
        assistant_service=object(),  # type: ignore[arg-type]
        onec_client=onec,  # type: ignore[arg-type]
        ollama_client=ollama,  # type: ignore[arg-type]
    )

    with pytest.raises(RuntimeError, match="close failed"):
        await resources.close()

    assert onec.closed is True
    assert ollama.closed is True
