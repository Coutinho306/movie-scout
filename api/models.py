"""Request/response models for the FastAPI backend.

Reuses ``agent.state.RecItem`` for citations — no duplicated rec schema.
"""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

from agent.state import RecItem
from ingestion.models import TasteProfile


class AskRequest(BaseModel):
    # Cap length: /ask fans out to embeddings + LLM turns, so an unbounded
    # query is a cost-amplification vector on a public endpoint.
    query: str = Field(min_length=1, max_length=2000)
    session_id: str | None = None
    taste_profile: TasteProfile | None = None
    # Franchise clarify round-trip fields (AC-4).
    # On the second /ask (after a needs_clarification=True response), the client
    # echoes back the original query + this answer.  clarification_answer carries
    # the user's free-text yes/no; franchise_sibling_ids echoes the list from
    # the first AskResponse so the server can apply the exclude filter without
    # re-running detection (stateless, single-clarify-turn contract).
    clarification_answer: str | None = Field(default=None, max_length=500)
    franchise_sibling_ids: list[int] = []


class TasteProfileResponse(BaseModel):
    """Response from POST /taste-profile — profile + resolution counts."""

    profile: TasteProfile
    resolved: int
    tmdb_miss: int
    out_of_corpus: int
    total_input: int


class AskResponse(BaseModel):
    run_id: UUID
    final_answer: str
    citations: list[RecItem]
    latency_ms: float
    cost_usd: float
    tool_calls: int
    # Franchise clarify round-trip fields (AC-4).
    # On a clarify-pause response (needs_clarification=True), the client must
    # display clarification_question to the user and echo franchise_sibling_ids
    # back on the follow-up /ask (stateless, no server session).
    needs_clarification: bool = False
    clarification_question: str | None = None
    franchise_sibling_ids: list[int] = []


class FeedbackRequest(BaseModel):
    run_id: UUID
    rating: Literal["up", "down"]
    comment: str | None = Field(default=None, max_length=2000)
