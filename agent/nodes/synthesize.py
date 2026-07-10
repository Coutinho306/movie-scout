"""Synthesize node: ranks pools into a final JSON recommendation list."""

from __future__ import annotations

import json
import logging
import re

from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI

from agent.config import AgentSettings
from agent.cost import usage_from_message
from agent.nodes import load_prompt
from agent.state import AgentState, RecItem

logger = logging.getLogger(__name__)


def _build_prompt(state: AgentState, settings: AgentSettings) -> str:
    template = load_prompt("synthesize")
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


def _extract_title_from_query(query: str) -> str | None:
    """Heuristically extract a film title from an inform-intent query.

    Looks for a quoted title first, then strips common question prefixes to
    isolate a bare title. Strips a trailing 4-digit year (e.g. "Obsession
    2026" -> "Obsession") since exact-title lookup matches the bare title
    payload field, not "title year". Returns None when extraction is
    uncertain — callers treat that as "no collision lookup needed".
    """
    # Quoted title: "What is the theme of 'Obsession'?" -> "Obsession"
    quoted = re.search(r'["‘’“”](.+?)["‘’“”]', query)
    if quoted:
        return quoted.group(1).strip()

    # Strip leading question phrases: "what is the theme of", "tell me about",
    # "who directed", "when was", "what year was", "where can I watch", etc.
    stripped = re.sub(
        r"^\s*(?:what\s+(?:is|are|was|were)\s+(?:the\s+)?"
        r"(?:(?:theme|plot|story|genre|cast|director|rating|year|overview|about|runtime|tagline|budget)\s+of\s+|about\s+)?|"
        r"who\s+(?:directed|starred\s+in|wrote|made|produced)\s+|"
        r"when\s+(?:was|is|did)\s+(?:the\s+)?(?:film\s+|movie\s+)?|"
        r"where\s+can\s+i\s+(?:watch|stream|find|see)\s+|"
        r"tell\s+me\s+about\s+(?:the\s+(?:film\s+|movie\s+))?|"
        r"(?:the\s+)?(?:film|movie)\s+)",
        "",
        query,
        flags=re.IGNORECASE,
    ).strip().rstrip("?.!")
    # Strip a trailing 4-digit year (the title payload field has no year).
    stripped = re.sub(r"\s+(?:19|20)\d{2}$", "", stripped).strip()
    # Only trust the result when it looks like a title: non-empty and not a
    # common pronoun / filler word (which indicate we didn't strip enough).
    if stripped and not re.match(r"^(?:it|that|this|the|a|an)\b", stripped, re.IGNORECASE):
        return stripped
    return None


def _supplement_collision_hits(
    rag_hits: list[dict],
    settings: AgentSettings,
    user_query: str = "",
) -> list[dict]:
    """Fetch the full exact-title collision set and merge any unseen films in.

    Checks every distinct title already in rag_hits (dense search may have
    surfaced one of several same-titled films), AND independently tries the
    title extracted from the user's own query — dense search can miss the
    named film entirely (non-deterministic LLM tool choice, or the title
    simply doesn't rank in top-k), in which case rag_hits has nothing to
    supplement from and the collision would otherwise never be found.

    This guarantees that when multiple films share a title (e.g. four films
    named "Obsession"), all of them reach the inform synthesis prompt so
    inform.md's disambiguation language can fire — independent of whether
    dense search happened to find any of them.

    Returns a new list (original is not mutated).
    """
    from retrieval.config import RetrievalSettings
    from retrieval.movies import find_by_exact_title

    retrieval_settings = RetrievalSettings()
    seen_ids: set[int] = {h.get("tmdb_id") for h in rag_hits if h.get("tmdb_id")}
    checked_titles: set[str] = set()
    supplemented = list(rag_hits)

    titles_to_check = [h.get("title", "") for h in rag_hits]
    query_title = _extract_title_from_query(user_query)
    if query_title:
        titles_to_check.append(query_title)

    for title in titles_to_check:
        if not title or title in checked_titles:
            continue
        checked_titles.add(title)
        try:
            collision_hits = find_by_exact_title(title, settings=retrieval_settings)
        except Exception:  # noqa: BLE001 — don't let a lookup error break synthesis
            continue
        for ch in collision_hits:
            if ch.tmdb_id not in seen_ids:
                supplemented.append(ch.model_dump())
                seen_ids.add(ch.tmdb_id)

    return supplemented


def synthesize_inform_node(state: AgentState, settings: AgentSettings) -> dict:
    """Answer an informational query with prose about one film; sets no recs.

    Used when the orchestrator classified ``intent == "inform"``. Grounds the
    answer on the film's TMDB ``overview`` (carried in ``rag_hits``) plus any web
    hits, and returns plain text — never a recommendation list.

    Before synthesis, ``_supplement_collision_hits`` extends ``rag_hits`` with
    any films sharing an exact title with already-found hits, so that
    inform.md's disambiguation language ("there are N films called X — did you
    mean the {{year}} one?") can fire when multiple films carry the same title.
    """
    cfg = settings
    llm = ChatOpenAI(model=cfg.model_agent, temperature=cfg.temperature)

    # Supplement rag_hits with the full exact-title collision set before synthesis.
    rag_hits = _supplement_collision_hits(
        state.get("rag_hits", []), settings, user_query=state["user_query"]
    )

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
