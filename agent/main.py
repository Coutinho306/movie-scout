"""LangGraph agent entrypoint for Movie Scout."""

from __future__ import annotations

import json
import logging
import os
import time

from agent.config import AgentSettings
from agent.graph import build_graph
from agent.state import AgentRunResult, AgentState, RecItem
from agent.tools.franchise import detect_franchise_ambiguity, resolve_clarification

logger = logging.getLogger(__name__)


def _tmdb_api_key() -> str:
    return os.environ.get("TMDB_API_KEY", "")


def run(user_query: str, settings: AgentSettings | None = None) -> AgentRunResult:
    """Run the agent end-to-end for a single query and return a structured result.

    Pre-graph franchise-ambiguity gate (AC-2, AC-7, AC-8):
    - If the request carries a ``clarification_answer`` (second call), skip
      detection, resolve include/exclude, and seed the initial state.
    - If no answer yet: run detection on seed-shaped queries. On ambiguity,
      return a clarify-pause AgentRunResult (empty recs, no graph.invoke).
    - If query is not seed-shaped or film has no corpus siblings: fall through
      to the normal graph run.
    """
    settings = settings or AgentSettings()

    # -----------------------------------------------------------------------
    # Phase A: handle a request that already carries a clarification answer.
    # Detection is NOT re-run (single clarify turn hard cap, AC-7).
    # -----------------------------------------------------------------------
    franchise_include: bool | None = None
    franchise_exclude_ids: list[int] = []

    if settings.clarification_answer is not None:
        # Resolve the free-text answer; default to exclude on unclear/None (AC-7)
        resolved = resolve_clarification(settings.clarification_answer)
        franchise_include = resolved if resolved is not None else False
        # Sibling ids are echoed back from the first call's AskResponse.
        # On the exclude path, we use them to filter out franchise siblings.
        sibling_ids = list(settings.franchise_sibling_ids or [])
        if not franchise_include:
            # False (exclude) or default-exclude (unclear answer → False)
            franchise_exclude_ids = sibling_ids

    else:
        # -----------------------------------------------------------------------
        # Phase B: first call — run detection (seed-shaped queries only).
        # -----------------------------------------------------------------------
        tmdb_key = _tmdb_api_key()
        if tmdb_key:
            try:
                ambiguity = detect_franchise_ambiguity(user_query, tmdb_api_key=tmdb_key)
            except Exception:  # noqa: BLE001 — detection failure must not block recs
                logger.exception("franchise detection failed; falling through to graph run")
                ambiguity = None
        else:
            ambiguity = None

        if ambiguity is not None:
            # Ambiguity detected and no answer yet → pause, ask, return early
            logger.info(
                json.dumps({
                    "step": "franchise_clarify_pause",
                    "seed_id": ambiguity.seed_id,
                    "seed_title": ambiguity.seed_title,
                    "collection": ambiguity.collection_name,
                    "sibling_ids": ambiguity.sibling_ids,
                })
            )
            return AgentRunResult(
                final_answer=ambiguity.question,
                citations=[],
                tool_calls=0,
                latency_ms=0.0,
                cost_usd=0.0,
                orchestrator_turns=0,
                rag_calls=0,
                web_calls=0,
                needs_clarification=True,
                clarification_question=ambiguity.question,
                franchise_sibling_ids=ambiguity.sibling_ids,
            )

    # -----------------------------------------------------------------------
    # Normal graph run (no ambiguity, or resolved answer)
    # -----------------------------------------------------------------------
    graph = build_graph(settings)

    start = time.time()
    initial_state: AgentState = {
        "user_query": user_query,
        "rewritten_query": None,
        "intent": None,
        "plan": [],
        "rag_hits": [],
        "web_hits": [],
        "messages": [],
        "final_answer": None,
        "recs": [],
        "orchestrator_turns": 0,
        "rag_calls": 0,
        "web_calls": 0,
        "cost_usd": 0.0,
        "token_count": 0,
        # Franchise clarify-turn fields (AC-3)
        "clarification_answer": settings.clarification_answer,
        "franchise_include": franchise_include,
        "franchise_exclude_ids": franchise_exclude_ids,
    }

    result = graph.invoke(initial_state)
    latency_ms = (time.time() - start) * 1000

    citations: list[RecItem] = []
    for item in result.get("recs", []):
        try:
            citations.append(RecItem(**item))
        except Exception:  # noqa: BLE001 — skip malformed rec
            continue

    log_line = {
        "step": "agent_run",
        "tokens": result.get("token_count", 0),
        "cost_usd": round(result.get("cost_usd", 0.0), 6),
        "orchestrator_turns": result.get("orchestrator_turns", 0),
        "rag_calls": result.get("rag_calls", 0),
        "web_calls": result.get("web_calls", 0),
        "latency_ms": round(latency_ms, 1),
    }
    logger.info(json.dumps(log_line))

    return AgentRunResult(
        final_answer=result.get("final_answer") or "No recommendation generated.",
        citations=citations,
        tool_calls=result.get("rag_calls", 0) + result.get("web_calls", 0),
        latency_ms=latency_ms,
        cost_usd=result.get("cost_usd", 0.0),
        orchestrator_turns=result.get("orchestrator_turns", 0),
        rag_calls=result.get("rag_calls", 0),
        web_calls=result.get("web_calls", 0),
        needs_clarification=False,
        clarification_question=None,
    )


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    query = " ".join(sys.argv[1:]) or "recommend something slow and meditative"
    run_result = run(query)
    print(run_result.final_answer)
