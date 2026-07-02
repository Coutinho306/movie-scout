"""Integration smoke test — requires live Qdrant + OpenAI + Tavily."""

from __future__ import annotations

import pytest

from agent.main import run


@pytest.mark.integration
def test_smoke_run() -> None:
    result = run("recommend something slow and meditative")

    assert result.final_answer
    assert len(result.citations) >= 1
    assert result.orchestrator_turns <= 4

    for rec in result.citations:
        assert rec.tmdb_id > 0
        assert rec.title
        assert rec.year > 0


@pytest.mark.integration
def test_smoke_inform_returns_prose_no_recs() -> None:
    result = run("What do you know about Arrival?")

    assert result.final_answer
    # inform mode answers with prose about the film — no recommendation citations.
    assert result.citations == []
    assert result.orchestrator_turns <= 4
