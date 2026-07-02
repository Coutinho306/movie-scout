"""LangGraph agent state (TypedDict) + Pydantic result/data models."""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel


class AgentState(TypedDict):
    """Shared mutable state threaded through the LangGraph nodes.

    TypedDict (not a Pydantic model) for LangGraph reducer compatibility.
    """

    user_query: str
    rewritten_query: str | None
    intent: str | None  # "recommend" | "inform" — classified once, first orchestrator turn
    plan: list[str]
    rag_hits: list[dict]  # serialized MovieHit dicts
    web_hits: list[dict]  # serialized WebHit dicts
    messages: Annotated[list[BaseMessage], add_messages]
    final_answer: str | None
    recs: list[dict]  # serialized RecItem dicts, set by synthesize
    orchestrator_turns: int
    rag_calls: int
    web_calls: int
    cost_usd: float
    token_count: int


class WebHit(BaseModel):
    url: str
    title: str
    content: str


class RecItem(BaseModel):
    tmdb_id: int
    title: str
    year: int
    why_for_you: str
    provider_hint: str | None = None


class AgentRunResult(BaseModel):
    final_answer: str
    citations: list[RecItem]
    tool_calls: int
    latency_ms: float
    cost_usd: float
    orchestrator_turns: int
    rag_calls: int
    web_calls: int
