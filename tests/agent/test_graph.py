"""Offline unit tests for graph routing and the rewrite node (no live LLM)."""

from __future__ import annotations

from agent.config import AgentSettings
from agent.graph import _route
from agent.nodes import rewrite as rewrite_mod
from agent.nodes.rewrite import rewrite_node

MAX_TURNS = 4


def _state(plan: list[str], turns: int, intent: str | None = None) -> dict:
    return {"plan": plan, "orchestrator_turns": turns, "intent": intent}


# --- _route (AC #4) -------------------------------------------------------


def test_route_rag_goes_through_rewrite() -> None:
    assert _route(_state(["rag"], 1), MAX_TURNS) == "rewrite"


def test_route_web() -> None:
    assert _route(_state(["web"], 1), MAX_TURNS) == "web_agent"


def test_route_synthesize() -> None:
    assert _route(_state(["synthesize"], 1), MAX_TURNS) == "synthesize"


def test_route_empty_plan_defaults_to_rewrite() -> None:
    # last_action defaults to "rag" -> rewrite.
    assert _route(_state([], 0), MAX_TURNS) == "rewrite"


def test_route_turn_cap_forces_synthesize() -> None:
    # At the cap, synthesize regardless of the picked action.
    assert _route(_state(["rag"], MAX_TURNS), MAX_TURNS) == "synthesize"


def test_route_repeat_guard_rag() -> None:
    assert _route(_state(["rag", "rag"], 2), MAX_TURNS) == "synthesize"


def test_route_repeat_guard_web() -> None:
    assert _route(_state(["web", "web"], 2), MAX_TURNS) == "synthesize"


def test_route_non_repeat_routes_normally() -> None:
    # web then rag is a legitimate sequence — not a back-to-back duplicate.
    assert _route(_state(["web", "rag"], 2), MAX_TURNS) == "rewrite"


# --- _route inform intent -------------------------------------------------


def test_route_inform_synthesize_goes_to_inform_node() -> None:
    assert _route(_state(["synthesize"], 1, intent="inform"), MAX_TURNS) == "synthesize_inform"


def test_route_inform_turn_cap_still_informs() -> None:
    assert _route(_state(["rag"], MAX_TURNS, intent="inform"), MAX_TURNS) == "synthesize_inform"


def test_route_inform_repeat_guard_still_informs() -> None:
    assert _route(_state(["rag", "rag"], 2, intent="inform"), MAX_TURNS) == "synthesize_inform"


def test_route_inform_still_runs_rag_first() -> None:
    # inform must retrieve the film before answering — rag action still routes to rewrite.
    assert _route(_state(["rag"], 1, intent="inform"), MAX_TURNS) == "rewrite"


# --- rewrite_node (AC #2, #3) --------------------------------------------


def test_rewrite_sets_query_when_enabled(monkeypatch) -> None:
    monkeypatch.setattr(rewrite_mod, "rewrite_query", lambda q: f"REW:{q}")
    settings = AgentSettings(query_rewrite=True)
    out = rewrite_node({"user_query": "slow films", "rewritten_query": None}, settings)
    assert out == {"rewritten_query": "REW:slow films"}


def test_rewrite_noop_when_disabled(monkeypatch) -> None:
    monkeypatch.setattr(
        rewrite_mod, "rewrite_query", lambda q: (_ for _ in ()).throw(AssertionError("called"))
    )
    settings = AgentSettings(query_rewrite=False)
    out = rewrite_node({"user_query": "slow films", "rewritten_query": None}, settings)
    assert out == {}


def test_rewrite_idempotent_when_already_set(monkeypatch) -> None:
    # Already rewritten -> skip the call (one rewrite per run, not per turn).
    monkeypatch.setattr(
        rewrite_mod, "rewrite_query", lambda q: (_ for _ in ()).throw(AssertionError("called"))
    )
    settings = AgentSettings(query_rewrite=True)
    out = rewrite_node({"user_query": "slow films", "rewritten_query": "already"}, settings)
    assert out == {}
