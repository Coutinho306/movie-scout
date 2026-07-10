"""API-level tests for the disambiguation clarify round-trip (0013 AC-2, AC-6, AC-7, AC-8, AC-10).

Uses the same stateless approach as test_franchise_api.py:
- Fresh rate limiter per test via monkeypatch.
- agent run function stubbed; no live TMDB, Qdrant, or LLM calls.

Tests:
- First /ask (ambiguous Obsession query) → needs_clarification=True + templated
  question + empty citations + empty franchise_sibling_ids (AC-2, AC-10).
- Second /ask with clarification_answer="the 1976 one" → normal single-film inform
  answer, needs_clarification=False, no re-ask (AC-6, AC-10).
- Unresolvable follow-up → AC-7 fallback answer (needs_clarification=False) (AC-7).
- Unique-title inform query → unchanged response shape (AC-8, AC-10).
- Request without clarification fields (backward compat) → still works (AC-10).
"""

from __future__ import annotations

from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient
from slowapi import Limiter

from agent.config import AgentSettings
from agent.state import AgentRunResult, RecItem
from api.dependencies import get_agent_run_fn, get_agent_settings, get_pg_pool
import api.fastapi_app as fastapi_app
from api.fastapi_app import create_app


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

# Simulated templated question for "Obsession" (4 films)
_OBSESSION_QUESTION = (
    "There are 4 films called Obsession: from 1943, 1976, 2015, 2026 — which one did you mean?"
)

_COLLISION_PAUSE_RESULT = AgentRunResult(
    final_answer=_OBSESSION_QUESTION,
    citations=[],
    tool_calls=0,
    latency_ms=3.0,
    cost_usd=0.0,
    orchestrator_turns=0,
    rag_calls=0,
    web_calls=0,
    needs_clarification=True,
    clarification_question=_OBSESSION_QUESTION,
    franchise_sibling_ids=[],  # collision pause carries no sibling ids
)

_OBSESSION_1976_RESULT = AgentRunResult(
    final_answer=(
        "Obsession (1976) is a psychological thriller directed by Brian De Palma. "
        "It stars Cliff Robertson as a man haunted by the disappearance of his wife."
    ),
    citations=[
        RecItem(tmdb_id=4780, title="Obsession", year=1976, why_for_you="Requested film."),
    ],
    tool_calls=1,
    latency_ms=320.0,
    cost_usd=0.002,
    orchestrator_turns=1,
    rag_calls=1,
    web_calls=0,
    needs_clarification=False,
    clarification_question=None,
    franchise_sibling_ids=[],
)

_FALLBACK_NEWEST_RESULT = AgentRunResult(
    final_answer=(
        "Obsession (2026) is a romantic thriller streaming on Netflix."
    ),
    citations=[
        RecItem(tmdb_id=1339713, title="Obsession", year=2026, why_for_you="Newest Obsession."),
    ],
    tool_calls=1,
    latency_ms=310.0,
    cost_usd=0.002,
    orchestrator_turns=1,
    rag_calls=1,
    web_calls=0,
    needs_clarification=False,
    clarification_question=None,
    franchise_sibling_ids=[],
)

_UNIQUE_TITLE_RESULT = AgentRunResult(
    final_answer="Fight Club (1999) was directed by David Fincher.",
    citations=[
        RecItem(tmdb_id=550, title="Fight Club", year=1999, why_for_you="Unique title."),
    ],
    tool_calls=1,
    latency_ms=280.0,
    cost_usd=0.002,
    orchestrator_turns=1,
    rag_calls=1,
    web_calls=0,
    needs_clarification=False,
    clarification_question=None,
    franchise_sibling_ids=[],
)


@contextmanager
def _fresh_client(agent_run_fn, monkeypatch):
    """Create a TestClient with a fresh rate limiter to avoid inter-test interference."""
    monkeypatch.setattr(
        fastapi_app, "limiter", Limiter(key_func=fastapi_app._client_key)
    )
    app = create_app()
    app.dependency_overrides[get_agent_run_fn] = lambda: agent_run_fn
    app.dependency_overrides[get_agent_settings] = lambda: AgentSettings()
    app.dependency_overrides[get_pg_pool] = lambda: None
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDisambiguationClarifyRoundTrip:
    def test_first_ask_ambiguous_obsession_returns_clarify_pause(self, monkeypatch) -> None:
        """First /ask with colliding-title → needs_clarification=True + templated question."""
        captured_settings: list[AgentSettings] = []

        def stub_run(query: str, settings: AgentSettings | None = None) -> AgentRunResult:
            if settings is not None:
                captured_settings.append(settings)
            return _COLLISION_PAUSE_RESULT

        with _fresh_client(stub_run, monkeypatch) as c:
            resp = c.post("/ask", json={"query": "When was Obsession released?"})

        assert resp.status_code == 200
        body = resp.json()

        # AC-2: pause response shape
        assert body["needs_clarification"] is True
        assert body["clarification_question"] is not None
        assert "Obsession" in body["clarification_question"]
        assert "1943" in body["clarification_question"]
        assert "1976" in body["clarification_question"]
        assert "2015" in body["clarification_question"]
        assert "2026" in body["clarification_question"]
        assert body["citations"] == []
        assert body["tool_calls"] == 0
        assert body["cost_usd"] == 0.0
        # Collision pause carries no franchise sibling ids (different mechanism)
        assert body["franchise_sibling_ids"] == []

    def test_second_ask_the_1976_one_returns_single_film_answer(self, monkeypatch) -> None:
        """Second /ask with clarification_answer='the 1976 one' → normal inform answer, no re-ask."""
        captured_settings: list[AgentSettings] = []

        def stub_run(query: str, settings: AgentSettings | None = None) -> AgentRunResult:
            if settings is not None:
                captured_settings.append(settings)
            return _OBSESSION_1976_RESULT

        with _fresh_client(stub_run, monkeypatch) as c:
            resp = c.post(
                "/ask",
                json={
                    "query": "When was Obsession released?",
                    "clarification_answer": "the 1976 one",
                },
            )

        assert resp.status_code == 200
        body = resp.json()

        # AC-6: second call resolves to single-film answer
        assert body["needs_clarification"] is False
        assert body["clarification_question"] is None
        assert len(body["citations"]) == 1
        assert body["citations"][0]["tmdb_id"] == 4780  # 1976 Obsession
        assert "1976" in body["final_answer"]

        # Verify clarification_answer was threaded to the agent run
        assert len(captured_settings) == 1
        assert captured_settings[0].clarification_answer == "the 1976 one"

    def test_unresolvable_follow_up_returns_fallback_answer(self, monkeypatch) -> None:
        """Unresolvable clarification_answer → AC-7 fallback answer (no re-ask, no crash)."""
        def stub_run(query: str, settings: AgentSettings | None = None) -> AgentRunResult:
            return _FALLBACK_NEWEST_RESULT

        with _fresh_client(stub_run, monkeypatch) as c:
            resp = c.post(
                "/ask",
                json={
                    "query": "When was Obsession released?",
                    "clarification_answer": "the 1990 one",  # out-of-tolerance
                },
            )

        assert resp.status_code == 200
        body = resp.json()

        # AC-7: fallback → normal answer (newest), no re-ask
        assert body["needs_clarification"] is False
        assert body["clarification_question"] is None
        assert len(body["citations"]) >= 1

    def test_unique_title_inform_query_unchanged_response(self, monkeypatch) -> None:
        """A unique-title inform query (no collision) returns normal response (AC-8)."""
        def stub_run(query: str, settings: AgentSettings | None = None) -> AgentRunResult:
            return _UNIQUE_TITLE_RESULT

        with _fresh_client(stub_run, monkeypatch) as c:
            resp = c.post("/ask", json={"query": "Who directed Fight Club?"})

        assert resp.status_code == 200
        body = resp.json()
        assert body["needs_clarification"] is False
        assert body["clarification_question"] is None
        assert len(body["citations"]) == 1
        assert body["citations"][0]["tmdb_id"] == 550

    def test_clarification_fields_not_required_backward_compat(self, monkeypatch) -> None:
        """Old clients without clarification fields still work (AC-10 regression)."""
        def stub_run(query: str, settings: AgentSettings | None = None) -> AgentRunResult:
            return _UNIQUE_TITLE_RESULT

        with _fresh_client(stub_run, monkeypatch) as c:
            # Old-style request — no clarification fields at all
            resp = c.post("/ask", json={"query": "tell me about Fight Club"})

        assert resp.status_code == 200
        body = resp.json()
        # New fields present but default to safe values
        assert "needs_clarification" in body
        assert "clarification_question" in body
        assert "franchise_sibling_ids" in body
        assert body["needs_clarification"] is False

    def test_clarification_answer_is_threaded_to_agent_settings(self, monkeypatch) -> None:
        """On the second call, clarification_answer arrives in AgentSettings (AC-10 plumbing)."""
        captured_settings: list[AgentSettings] = []

        def stub_run(query: str, settings: AgentSettings | None = None) -> AgentRunResult:
            if settings is not None:
                captured_settings.append(settings)
            return _OBSESSION_1976_RESULT

        with _fresh_client(stub_run, monkeypatch) as c:
            c.post(
                "/ask",
                json={
                    "query": "Tell me about Obsession",
                    "clarification_answer": "1976",
                },
            )

        assert len(captured_settings) == 1
        assert captured_settings[0].clarification_answer == "1976"
