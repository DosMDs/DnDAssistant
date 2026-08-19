"""Typed, provider-neutral models owned by the agent layer."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from ..models import ChatMessage


class AgentLimits(BaseModel):
    """Hard limits for one transient run.

    One iteration is one completed call to ``ModelProvider.complete``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_iterations: int = Field(default=8, gt=0)
    max_total_tool_calls: int = Field(default=16, gt=0)


class AgentResult(BaseModel):
    """Successful final outcome and the complete in-memory transcript."""

    model_config = ConfigDict(extra="forbid")

    final_message: ChatMessage
    messages: list[ChatMessage]
    iterations: int = Field(ge=1)
    total_tool_calls: int = Field(ge=0)
