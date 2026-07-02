"""Offline unit tests for orchestrator intent resolution (no live LLM)."""

from __future__ import annotations

from agent.nodes.orchestrator import _resolve_intent


def test_intent_defaults_to_recommend_when_unset_and_missing() -> None:
    assert _resolve_intent(None, None) == "recommend"


def test_intent_takes_first_turn_classification() -> None:
    assert _resolve_intent(None, "inform") == "inform"


def test_intent_invalid_parsed_falls_back_to_recommend() -> None:
    assert _resolve_intent(None, "banana") == "recommend"


def test_intent_is_sticky_ignores_later_turns() -> None:
    # Once set to inform, a later turn parsing "recommend" cannot flip it.
    assert _resolve_intent("inform", "recommend") == "inform"


def test_intent_sticky_survives_later_malformed() -> None:
    assert _resolve_intent("inform", None) == "inform"
