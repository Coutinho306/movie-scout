"""Synthesize node: ranks pools into a final JSON recommendation list."""

from __future__ import annotations

import json
import logging

from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI

from agent.config import AgentSettings
from agent.cost import usage_from_message
from agent.nodes import load_prompt
from agent.state import AgentState, RecItem

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Low-signal output gate
# ---------------------------------------------------------------------------

# Minimum cosine-similarity score for a RAG hit to be considered confident.
# Calibrated 2026-07-23 against golden-set queries vs item-4a gibberish probes:
#   - Dense golden queries:  P10=0.445, min top-1=0.443, mean=0.528
#   - Gibberish top-1 max:   0.337  (clean separation margin ~0.06)
# A floor of 0.40 keeps all golden top-1 hits above it and rejects all
# tested gibberish probes.
#
# In hybrid/RRF mode, MovieHit.score is a rank-fraction (1/rank), not a cosine
# distance, and cannot separate gibberish from real queries.  The gate therefore
# reads MovieHit.dense_score (the raw cosine vs query_vec computed client-side
# in retrieval/movies.py from the vectors already returned by with_vectors=True)
# when hits are RRF-shaped.  dense_score is calibrated on the same cosine scale,
# so the same 0.40 floor applies unchanged (confirmed by AC-7 spot-check in
# specs/0025-hybrid-rrf-score-floor-fix/STATUS.md).
SCORE_FLOOR: float = 0.40

_DEFLECTION_ANSWER = (
    "I couldn't find films that confidently match that. "
    "Try rephrasing, or give me a title, genre, or vibe to anchor on."
)


def _is_rrf_score(score: float) -> bool:
    """Return True if score looks like an RRF fraction (1/k for small k)."""
    if score <= 0:
        return False
    for k in range(1, 100):
        if abs(score - 1.0 / k) < 1e-4:
            return True
    return False


def _hits_are_rrf_mode(rag_hits: list[dict]) -> bool:
    """Return True if the hit set appears to come from hybrid/RRF retrieval."""
    if not rag_hits:
        return False
    return all(_is_rrf_score(h.get("score", 0.0)) for h in rag_hits)


def _build_prompt(state: AgentState, settings: AgentSettings) -> str:
    prompt_name = "synthesize" if settings.prompt_variant == "v1" else f"synthesize_{settings.prompt_variant}"
    template = load_prompt(prompt_name)
    rag_hits = json.dumps(state.get("rag_hits", []), ensure_ascii=False)
    web_hits = json.dumps(state.get("web_hits", []), ensure_ascii=False)
    profile = settings.taste_profile
    top_films = ", ".join(profile.top_films) if profile and profile.top_films else "none"
    return template.format(
        rag_hits=rag_hits,
        web_hits=web_hits,
        user_query=state["user_query"],
        taste_top_films=top_films,
    )


def _format_answer(recs: list[RecItem]) -> str:
    if not recs:
        return "No recommendation generated."
    blocks: list[str] = []
    for i, rec in enumerate(recs, start=1):
        header = f"**{i}. {rec.title}** ({rec.year})"
        if rec.provider_hint:
            header += f" — _{rec.provider_hint}_"
        blocks.append(header)
        blocks.append(rec.why_for_you)
        blocks.append("")
    return "\n".join(blocks).rstrip()


def synthesize_node(state: AgentState, settings: AgentSettings) -> dict:
    """Produce the final ranked recommendation; sets ``final_answer``."""
    cfg = settings
    use_json_mode = cfg.model_agent not in cfg.reasoning_models

    llm_kwargs: dict = {"model": cfg.model_agent, "temperature": cfg.temperature}
    if use_json_mode:
        # json_object mode requires the word "json" in the prompt — the template has it.
        llm_kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}

    llm = ChatOpenAI(**llm_kwargs)
    parser = JsonOutputParser()

    prompt = _build_prompt(state, settings)
    response = llm.invoke(prompt)

    rag_hits: list[dict] = state.get("rag_hits", [])
    rif_mode = _hits_are_rrf_mode(rag_hits)

    # valid_ids: tmdb_ids from real RAG hits (blocks LLM-hallucinated ids)
    valid_ids = {h.get("tmdb_id") for h in rag_hits}

    # above_floor_ids: tmdb_ids whose hit cleared SCORE_FLOOR.
    # In dense mode, floor is checked against the cosine `score`.
    # In hybrid/RRF mode, `score` is a rank-fraction and cannot discriminate;
    # `dense_score` (raw cosine vs query_vec, computed client-side in
    # retrieval/movies.py) is used instead — same cosine scale, same floor.
    score_key = "dense_score" if rif_mode else "score"
    if rag_hits:
        above_floor_ids = {
            h.get("tmdb_id")
            for h in rag_hits
            if h.get(score_key, 0.0) >= SCORE_FLOOR
        }
    else:
        above_floor_ids = valid_ids

    # Low-signal gate: if no hits cleared the floor, deflect immediately.
    if rag_hits and not above_floor_ids:
        logger.info(
            "synthesize: all %d RAG hits below SCORE_FLOOR=%.2f — deflecting",
            len(rag_hits),
            SCORE_FLOOR,
        )
        tokens, cost = usage_from_message(response, cfg.model_agent)
        return {
            "final_answer": _DEFLECTION_ANSWER,
            "recs": [],
            "token_count": state.get("token_count", 0) + tokens,
            "cost_usd": state.get("cost_usd", 0.0) + cost,
        }

    recs: list[RecItem] = []
    try:
        parsed = parser.invoke(response)
        # json_object mode may wrap the array under a key; normalize to a list.
        if isinstance(parsed, dict):
            parsed = next((v for v in parsed.values() if isinstance(v, list)), [parsed])
        for item in parsed:
            try:
                rec = RecItem(**item)
            except Exception:  # noqa: BLE001 — skip malformed entries
                continue
            # Keep only hits that are both real (in valid_ids) AND above floor
            effective_valid = above_floor_ids if above_floor_ids != valid_ids else valid_ids
            if effective_valid and rec.tmdb_id not in effective_valid:
                continue
            recs.append(rec)
    except Exception as exc:  # noqa: BLE001 — bad JSON yields empty recs
        logger.warning("Synthesize JSON parse failed: %s", exc)

    tokens, cost = usage_from_message(response, cfg.model_agent)

    return {
        "final_answer": _format_answer(recs),
        "recs": [r.model_dump() for r in recs],
        "token_count": state.get("token_count", 0) + tokens,
        "cost_usd": state.get("cost_usd", 0.0) + cost,
    }



def synthesize_inform_node(state: AgentState, settings: AgentSettings) -> dict:
    """Answer an informational query with prose about one film; sets no recs.

    Used when the orchestrator classified ``intent == "inform"``. Grounds the
    answer on the film's TMDB ``overview`` (carried in ``rag_hits``) plus any web
    hits, and returns plain text — never a recommendation list.

    When ``resolved_inform_tmdb_id`` is set (a disambiguation second turn, 0013
    AC-6), it fetches that single film by point id and uses it as the sole
    rag_hits entry, so the answer is about exactly the resolved film only. The
    collision supplement path is NOT run in that case — the id is already
    resolved and no collision can re-fire.

    Otherwise (no resolved id), passes rag_hits through as-is — the
    collision disambiguation question is now produced deterministically
    pre-graph (0013 AC-9) so the LLM no longer emits it.
    """
    cfg = settings
    llm = ChatOpenAI(model=cfg.model_agent, temperature=cfg.temperature)

    resolved_id: int | None = state.get("resolved_inform_tmdb_id")  # type: ignore[assignment]

    if resolved_id is not None:
        # Disambiguation second turn: fetch the single resolved film.
        from agent.tools.disambiguation import fetch_film_by_tmdb_id
        from retrieval.config import RetrievalSettings

        hit = fetch_film_by_tmdb_id(resolved_id, settings=RetrievalSettings())
        rag_hits = [hit] if hit is not None else state.get("rag_hits", [])
    else:
        # Normal path: pass rag_hits through unchanged.
        # (Collision supplement removed: 0013 pre-graph gate now handles
        #  disambiguation before the graph runs — AC-9 single source of truth.)
        rag_hits = state.get("rag_hits", [])

    template = load_prompt("inform")
    prompt = template.format(
        user_query=state["user_query"],
        rag_hits=json.dumps(rag_hits, ensure_ascii=False),
        web_hits=json.dumps(state.get("web_hits", []), ensure_ascii=False),
    )
    response = llm.invoke(prompt)
    answer = (response.content or "").strip() if isinstance(response.content, str) else ""

    tokens, cost = usage_from_message(response, cfg.model_agent)

    return {
        "final_answer": answer or "No information found for that film.",
        "recs": [],
        "token_count": state.get("token_count", 0) + tokens,
        "cost_usd": state.get("cost_usd", 0.0) + cost,
    }
