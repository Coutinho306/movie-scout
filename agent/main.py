"""LangGraph agent entrypoint for Movie Scout."""

from __future__ import annotations

import json
import logging
import os
import time

from agent.config import AgentSettings
from agent.graph import build_graph
from agent.state import AgentRunResult, AgentState, RecItem
from agent.tools.disambiguation import (
    build_collision_question,
    detect_title_collision,
    fetch_film_by_tmdb_id,
    resolve_year_reference,
)
from agent.tools.franchise import detect_franchise_ambiguity, resolve_clarification

logger = logging.getLogger(__name__)


def _tmdb_api_key() -> str:
    return os.environ.get("TMDB_API_KEY", "")


def _retrieval_settings() -> object:
    """Return a RetrievalSettings instance (lazy import — avoids boot-time cost)."""
    from retrieval.config import RetrievalSettings

    return RetrievalSettings()


def run(user_query: str, settings: AgentSettings | None = None) -> AgentRunResult:
    """Run the agent end-to-end for a single query and return a structured result.

    Pre-graph gate (two parallel branches, run in order):

    Branch 1 — Title-collision disambiguation (0013, AC-2, AC-7, AC-8):
      - First call (no clarification_answer): run detect_title_collision on the
        query. On collision → return a clarify-pause AgentRunResult with a
        templated question listing candidate years; no graph.invoke.
      - Second call (clarification_answer present): re-detect the collision
        deterministically from the echoed original query; resolve the year
        reference; seed resolved_inform_tmdb_id. On None (unclear/out-of-tol.)
        apply the AC-7 default (newest candidate). One-turn hard cap enforced.

    Branch 2 — Franchise ambiguity (0012, AC-2, AC-7, AC-8):
      - First call: run detect_franchise_ambiguity on seed-shaped queries.
        On ambiguity → clarify-pause. Only reached if Branch 1 did not fire.
      - Second call: resolve franchise include/exclude from clarification_answer
        and sibling ids. Detection not re-run (hard cap).

    Both branches share the same clarification_answer field and the same
    pre-graph gate seam — no second mechanism introduced.
    """
    settings = settings or AgentSettings()

    # State fields threaded into the graph initial state.
    franchise_include: bool | None = None
    franchise_exclude_ids: list[int] = []
    resolved_inform_tmdb_id: int | None = None

    if settings.clarification_answer is not None:
        # -----------------------------------------------------------------------
        # Phase A: second call — clarification_answer is present.
        # Hard cap: detection is NOT re-run for franchise (AC-7).
        # BUT: for collision disambiguation, we re-detect deterministically to
        # distinguish "this answer is for a collision turn" from "this answer is
        # for a franchise turn" without any extra state field.  Detection is
        # free (cheap Qdrant scroll, no LLM, no TMDB call) and deterministic.
        # -----------------------------------------------------------------------
        try:
            retrieval_cfg = _retrieval_settings()
            collision = detect_title_collision(user_query, settings=retrieval_cfg)
        except Exception:  # noqa: BLE001 — detection failure must not block the run
            logger.exception("collision re-detection on second call failed; ignoring")
            collision = None

        if collision is not None:
            # This is a disambiguation second turn.  Resolve the year reference.
            tmdb_id = resolve_year_reference(
                settings.clarification_answer, collision.candidates
            )
            if tmdb_id is None:
                # AC-7 fallback: unclear or out-of-tolerance year → newest candidate
                newest = max(collision.candidates, key=lambda c: c.year)
                tmdb_id = newest.tmdb_id
                logger.info(
                    json.dumps({
                        "step": "collision_fallback_newest",
                        "query": user_query,
                        "answer": settings.clarification_answer,
                        "tmdb_id": tmdb_id,
                    })
                )
            else:
                logger.info(
                    json.dumps({
                        "step": "collision_resolved",
                        "query": user_query,
                        "answer": settings.clarification_answer,
                        "tmdb_id": tmdb_id,
                    })
                )
            resolved_inform_tmdb_id = tmdb_id
        else:
            # Not a collision turn: resolve as franchise (0012 Phase A).
            resolved = resolve_clarification(settings.clarification_answer)
            franchise_include = resolved if resolved is not None else False
            sibling_ids = list(settings.franchise_sibling_ids or [])
            if not franchise_include:
                franchise_exclude_ids = sibling_ids

    else:
        # -----------------------------------------------------------------------
        # Phase B: first call — run detection.
        # Branch 1: title-collision detection (inform-shaped queries).
        # Branch 2: franchise-ambiguity detection (seed-shaped queries).
        # -----------------------------------------------------------------------

        # Branch 1 — collision detection (no TMDB needed, no LLM, cheap scroll)
        try:
            retrieval_cfg = _retrieval_settings()
            collision = detect_title_collision(user_query, settings=retrieval_cfg)
        except Exception:  # noqa: BLE001 — detection failure must not block recs
            logger.exception("collision detection failed; falling through to graph run")
            collision = None

        if collision is not None:
            # Collision detected, no answer yet → pause with templated question.
            question = build_collision_question(collision)
            logger.info(
                json.dumps({
                    "step": "collision_clarify_pause",
                    "title": collision.title,
                    "years": [c.year for c in collision.candidates],
                })
            )
            return AgentRunResult(
                final_answer=question,
                citations=[],
                tool_calls=0,
                latency_ms=0.0,
                cost_usd=0.0,
                orchestrator_turns=0,
                rag_calls=0,
                web_calls=0,
                needs_clarification=True,
                clarification_question=question,
            )

        # Branch 2 — franchise detection (only reached if no collision fired)
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
            # Franchise ambiguity detected → pause.
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
    # Normal graph run (no ambiguity / no collision, or resolved second call)
    # When resolved_inform_tmdb_id is set, synthesize_inform_node will answer
    # about that single film only (AC-6).
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
        # Franchise clarify-turn fields (0012)
        "clarification_answer": settings.clarification_answer,
        "franchise_include": franchise_include,
        "franchise_exclude_ids": franchise_exclude_ids,
        # Disambiguation clarify-turn field (0013)
        "resolved_inform_tmdb_id": resolved_inform_tmdb_id,
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
