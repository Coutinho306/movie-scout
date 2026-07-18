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

    recs: list[RecItem] = []
    valid_ids = {h.get("tmdb_id") for h in state.get("rag_hits", [])}
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
            if valid_ids and rec.tmdb_id not in valid_ids:
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
