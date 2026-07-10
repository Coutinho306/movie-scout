"""RAG worker node: a ReAct agent over the TMDB retrieval tools."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from agent.config import AgentSettings
from agent.cost import usage_from_messages
from agent.nodes import load_prompt
from agent.state import AgentState
from agent.tools.taste_matcher import match_taste_tool
from agent.tools.tmdb_providers import get_providers
from agent.tools.vector_search_movies import search_movies_tool
from agent.tools.vector_search_reviews import search_reviews_tool

if TYPE_CHECKING:
    from ingestion.models import TasteProfile

logger = logging.getLogger(__name__)


def _build_rag_tools(
    collected: list[dict],
    region: str,
    top_k: int = 10,
    taste_profile: "TasteProfile | None" = None,
    franchise_exclude_ids: list[int] | None = None,
) -> list:
    """Build ReAct tools bound to a run-local ``collected`` list for hit capture.

    ``taste_profile`` is the per-request profile (or None for cold start).
    ``franchise_exclude_ids`` is the list of franchise sibling tmdb_ids to
    exclude from the seed-similarity retrieval path (AC-6). When None or empty,
    no extra exclusion is applied (today's behaviour).
    """
    from ingestion.models import TasteProfile  # local import avoids circular at module level

    from retrieval.models import MovieFilters  # local import

    # Build the MovieFilters for seed-similarity calls (exclude path).
    # The include path adds no filter — sequels just aren't excluded from ranking.
    _seed_filters: MovieFilters | None = (
        MovieFilters(exclude_tmdb_ids=set(franchise_exclude_ids))
        if franchise_exclude_ids
        else None
    )

    @tool
    def search_movies(query: str, k: int = top_k) -> list[dict]:
        """Search the TMDB movie collection by semantic similarity. Returns movie dicts with tmdb_id, title, year, overview, genres."""
        hits = search_movies_tool(query, k=k, filters=_seed_filters)
        # vector is excluded at the model level (MovieHit.vector Field(exclude=True))
        dicts = [h.model_dump() for h in hits]
        seen = {d["tmdb_id"] for d in collected}
        for d in dicts:
            if d["tmdb_id"] not in seen:
                collected.append(d)
                seen.add(d["tmdb_id"])
        return dicts

    @tool
    def search_reviews(query: str, k: int = 10) -> list[dict]:
        """Search movie reviews by semantic similarity for deeper context. Returns review chunks with tmdb_id, title, author, text."""
        hits = search_reviews_tool(query, k=k)
        return [h.model_dump() for h in hits]

    @tool
    def match_taste(tmdb_ids: list[int]) -> list[dict]:
        """Score already-found movie candidates against the user's taste profile. Pass tmdb_ids of collected movies; returns them with taste_score and blended_score."""
        from retrieval.models import MovieHit

        subset = [MovieHit(**d) for d in collected if d["tmdb_id"] in set(tmdb_ids)]
        # profile=None → cold start (retrieval order, taste ignored)
        scored = match_taste_tool(subset, profile=taste_profile)
        scored_by_id = {h.tmdb_id: h.model_dump() for h in scored}
        for d in collected:
            if d["tmdb_id"] in scored_by_id:
                d.update(scored_by_id[d["tmdb_id"]])
        return list(scored_by_id.values())

    @tool
    def tmdb_lookup_providers(tmdb_id: int) -> list[str]:
        """Look up which streaming services offer a film by tmdb_id (region-specific flatrate providers).

        Writes the result onto the matching entry in the collected hits (as
        "providers") so it reaches synthesis/inform — call search_movies for
        this tmdb_id first if it isn't already in the collected hits.
        """
        providers = get_providers(tmdb_id, region=region)
        for d in collected:
            if d["tmdb_id"] == tmdb_id:
                d["providers"] = providers
        return providers

    return [search_movies, search_reviews, match_taste, tmdb_lookup_providers]


def build_rag_agent(
    settings: AgentSettings,
    collected: list[dict],
    franchise_exclude_ids: list[int] | None = None,
):
    """Construct a ReAct agent whose tools append hits to ``collected``."""
    llm = ChatOpenAI(model=settings.model_agent, temperature=settings.temperature)
    tools = _build_rag_tools(
        collected,
        settings.watch_region,
        top_k=settings.top_k,
        taste_profile=settings.taste_profile,
        franchise_exclude_ids=franchise_exclude_ids,
    )
    return create_react_agent(llm, tools=tools, prompt=load_prompt("rag_system"))


def rag_node(state: AgentState, settings: AgentSettings) -> dict:
    """Run the RAG ReAct agent, merge captured hits into state, track usage."""
    collected: list[dict] = list(state.get("rag_hits", []))
    # Read franchise exclude ids from state (set by the pre-graph gate on the
    # second /ask call when the user chose to exclude franchise siblings, AC-6).
    franchise_exclude_ids: list[int] = list(state.get("franchise_exclude_ids") or [])
    agent = build_rag_agent(settings, collected, franchise_exclude_ids=franchise_exclude_ids or None)

    query = state.get("rewritten_query") or state["user_query"]
    result = agent.invoke({"messages": [HumanMessage(content=query)]})

    messages = result.get("messages", [])
    tokens, cost = usage_from_messages(messages, settings.model_agent)

    return {
        "rag_hits": collected,
        "rag_calls": state.get("rag_calls", 0) + 1,
        "token_count": state.get("token_count", 0) + tokens,
        "cost_usd": state.get("cost_usd", 0.0) + cost,
    }
