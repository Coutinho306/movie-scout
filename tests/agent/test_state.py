"""Unit tests for agent state/data models — no network required."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent.state import AgentRunResult, AgentState, RecItem, WebHit


def test_agent_state_is_typeddict_with_expected_keys() -> None:
    keys = set(AgentState.__annotations__)
    assert {
        "user_query",
        "rewritten_query",
        "plan",
        "rag_hits",
        "web_hits",
        "messages",
        "final_answer",
        "orchestrator_turns",
        "rag_calls",
        "web_calls",
        "cost_usd",
        "token_count",
    } <= keys


def test_web_hit_validates() -> None:
    hit = WebHit(url="https://example.com", title="A film", content="snippet")
    assert hit.url == "https://example.com"


def test_rec_item_defaults_and_validation() -> None:
    rec = RecItem(tmdb_id=42, title="Stalker", year=1979, why_for_you="slow + meditative")
    assert rec.provider_hint is None
    assert rec.tmdb_id == 42

    with pytest.raises(ValidationError):
        RecItem(tmdb_id="not-an-int", title="x", year=2000, why_for_you="y")


def test_agent_run_result_round_trips() -> None:
    rec = RecItem(tmdb_id=1, title="t", year=2000, why_for_you="w", provider_hint="Netflix BR")
    result = AgentRunResult(
        final_answer="answer",
        citations=[rec],
        tool_calls=2,
        latency_ms=123.4,
        cost_usd=0.001,
        orchestrator_turns=3,
        rag_calls=1,
        web_calls=1,
    )
    dumped = result.model_dump()
    assert dumped["citations"][0]["tmdb_id"] == 1
    assert dumped["tool_calls"] == 2
