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
    # Franchise clarify-turn fields (AC-3)
    clarification_answer: str | None  # echoed back from the client on the second call
    franchise_include: bool | None    # tri-state: None=undecided, True=include, False=exclude
    franchise_exclude_ids: list[int]  # sibling corpus tmdb_ids to drop when exclude=True
    # Disambiguation clarify-turn field (0013-disambiguation-followup-turn, AC-3)
    # When non-None, synthesize_inform_node answers about this single resolved film only.
    resolved_inform_tmdb_id: int | None


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
    retrieved_tmdb_ids: list[int] = []  # raw RAG hit ids, for hallucination-rate scoring
    tool_calls: int
    latency_ms: float
    cost_usd: float
    orchestrator_turns: int
    rag_calls: int
    web_calls: int
    # Franchise clarify-turn fields (AC-3)
    needs_clarification: bool = False
    clarification_question: str | None = None
    # Sibling ids returned by the clarify-pause response so the client can
    # echo them back on the second /ask for the exclude filter (AC-6).
    franchise_sibling_ids: list[int] = []
