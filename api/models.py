"""Request/response models for the FastAPI backend.

Reuses ``agent.state.RecItem`` for citations — no duplicated rec schema.
"""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

from agent.state import RecItem


class AskRequest(BaseModel):
    # Cap length: /ask fans out to embeddings + LLM turns, so an unbounded
    # query is a cost-amplification vector on a public endpoint.
    query: str = Field(min_length=1, max_length=2000)
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
    comment: str | None = Field(default=None, max_length=2000)
