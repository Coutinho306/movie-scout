"""Orchestrator node: a plain JSON-mode LLM call that picks the next action."""

from __future__ import annotations

import logging

from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI

from agent.config import AgentSettings
from agent.cost import usage_from_message
from agent.nodes import load_prompt
from agent.state import AgentState

logger = logging.getLogger(__name__)

_VALID_ACTIONS = {"rag", "web", "synthesize"}


def _build_prompt(state: AgentState, cfg: AgentSettings) -> str:
    template = load_prompt("orchestrator")
    return template.format(
        rag_hits_count=len(state.get("rag_hits", [])),
        web_hits_count=len(state.get("web_hits", [])),
        orchestrator_turns=state.get("orchestrator_turns", 0),
        max_turns=cfg.max_orchestrator_turns,
        user_query=state["user_query"],
    )


def orchestrator_node(state: AgentState, settings: AgentSettings) -> dict:
    """Decide the next action and return state updates (turn count, plan, cost)."""
    cfg = settings
    use_json_mode = cfg.model_orchestrator not in cfg.reasoning_models

    llm_kwargs: dict = {"model": cfg.model_orchestrator, "temperature": 0}
    if use_json_mode:
        llm_kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}

    llm = ChatOpenAI(**llm_kwargs)
    parser = JsonOutputParser()

    prompt = _build_prompt(state, cfg)
    response = llm.invoke(prompt)

    try:
        action_dict = parser.invoke(response)
        action = action_dict.get("action", "synthesize")
    except Exception as exc:  # noqa: BLE001 — malformed JSON falls back to synthesize
        logger.warning("Orchestrator JSON parse failed: %s", exc)
        action = "synthesize"

    if action not in _VALID_ACTIONS:
        action = "synthesize"

    tokens, cost = usage_from_message(response, cfg.model_orchestrator)

    return {
        "orchestrator_turns": state.get("orchestrator_turns", 0) + 1,
        "plan": state.get("plan", []) + [action],
        "token_count": state.get("token_count", 0) + tokens,
        "cost_usd": state.get("cost_usd", 0.0) + cost,
    }
