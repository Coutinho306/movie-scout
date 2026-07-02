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


def _build_prompt(state: AgentState) -> str:
    template = load_prompt("synthesize")
    rag_hits = json.dumps(state.get("rag_hits", []), ensure_ascii=False)
    web_hits = json.dumps(state.get("web_hits", []), ensure_ascii=False)
    return template.format(
        rag_hits=rag_hits,
        web_hits=web_hits,
        user_query=state["user_query"],
    )


def _format_answer(recs: list[RecItem]) -> str:
    if not recs:
        return "No recommendation generated."
    lines: list[str] = []
    for i, rec in enumerate(recs, start=1):
        header = f"{i}. {rec.title} ({rec.year}) [tmdb:{rec.tmdb_id}]"
        if rec.provider_hint:
            header += f" — {rec.provider_hint}"
        lines.append(header)
        lines.append(f"   {rec.why_for_you}")
    return "\n".join(lines)


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

    prompt = _build_prompt(state)
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
    """
    cfg = settings
    llm = ChatOpenAI(model=cfg.model_agent, temperature=cfg.temperature)

    template = load_prompt("inform")
    prompt = template.format(
        user_query=state["user_query"],
        rag_hits=json.dumps(state.get("rag_hits", []), ensure_ascii=False),
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
