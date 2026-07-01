"""Request/response models for the FastAPI backend.

Reuses ``agent.state.RecItem`` for citations — no duplicated rec schema.
"""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel

from agent.state import RecItem


class AskRequest(BaseModel):
    query: str
    session_id: str | None = None


class AskResponse(BaseModel):
    run_id: UUID
    final_answer: str
    citations: list[RecItem]
    latency_ms: float
    cost_usd: float
    tool_calls: int


class FeedbackRequest(BaseModel):
    run_id: UUID
    rating: Literal["up", "down"]
    comment: str | None = None
