"""Rewrite node: reformulate the user query once before the first RAG retrieval."""

from __future__ import annotations

from agent.config import AgentSettings
from agent.state import AgentState
from retrieval.rewrite import rewrite_query


def rewrite_node(state: AgentState, settings: AgentSettings) -> dict:
    """Set ``rewritten_query`` from the user query. No-op if disabled or already set.

    Idempotent: once ``rewritten_query`` is populated, later passes skip the call,
    so rewriting costs one LLM call per run rather than per orchestrator turn.
    """
    if not settings.query_rewrite or state.get("rewritten_query"):
        return {}
    return {"rewritten_query": rewrite_query(state["user_query"])}
