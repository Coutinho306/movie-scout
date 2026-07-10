"""API-level tests for franchise clarify round-trip — stateless, no live calls.

Covers AC-3, AC-4, AC-6: first /ask → clarify-pause; second /ask with answer →
normal recs; non-franchise query → unchanged response shape.
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
from api.config import ApiSettings


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

KNIVES_OUT_TMDB_ID = 546554
GLASS_ONION_TMDB_ID = 764426

_CLARIFY_RESULT = AgentRunResult(
    final_answer=(
        '"Knives Out" is part of the Knives Out Collection — do you want those included, '
        "or just films with a similar mystery/comedy vibe? (yes / no)"
    ),
    citations=[],
    tool_calls=0,
    latency_ms=12.0,
    cost_usd=0.0,
    orchestrator_turns=0,
    rag_calls=0,
    web_calls=0,
    needs_clarification=True,
    clarification_question=(
        '"Knives Out" is part of the Knives Out Collection — do you want those included, '
        "or just films with a similar mystery/comedy vibe? (yes / no)"
    ),
    franchise_sibling_ids=[GLASS_ONION_TMDB_ID],
)

_NORMAL_RESULT = AgentRunResult(
    final_answer="Here are some films with a similar mystery vibe.",
    citations=[
        RecItem(tmdb_id=999, title="Clue", year=1985, why_for_you="Classic murder mystery."),
    ],
    tool_calls=2,
    latency_ms=450.0,
    cost_usd=0.003,
    orchestrator_turns=1,
    rag_calls=2,
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

class TestFranchiseClarifyRoundTrip:
    def test_first_ask_returns_clarify_pause_empty_citations(self, monkeypatch) -> None:
        """First /ask with ambiguous query → needs_clarification=True + empty citations."""
        captured_settings: list[AgentSettings] = []

        def stub_run(query: str, settings: AgentSettings | None = None) -> AgentRunResult:
            if settings is not None:
                captured_settings.append(settings)
            return _CLARIFY_RESULT

        with _fresh_client(stub_run, monkeypatch) as c:
            resp = c.post("/ask", json={"query": "films like Knives Out"})

        assert resp.status_code == 200
        body = resp.json()

        assert body["needs_clarification"] is True
        assert body["clarification_question"] is not None
        assert "Knives Out" in body["clarification_question"]
        assert body["citations"] == []
        assert body["tool_calls"] == 0
        assert GLASS_ONION_TMDB_ID in body["franchise_sibling_ids"]

    def test_second_ask_with_exclude_answer_returns_normal_recs(self, monkeypatch) -> None:
        """Second /ask with clarification_answer → needs_clarification=False + recs."""
        captured_settings: list[AgentSettings] = []

        def stub_run(query: str, settings: AgentSettings | None = None) -> AgentRunResult:
            if settings is not None:
                captured_settings.append(settings)
            return _NORMAL_RESULT

        with _fresh_client(stub_run, monkeypatch) as c:
            resp = c.post(
                "/ask",
                json={
                    "query": "films like Knives Out",
                    "clarification_answer": "no",
                    "franchise_sibling_ids": [GLASS_ONION_TMDB_ID],
                },
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["needs_clarification"] is False
        assert body["clarification_question"] is None
        assert len(body["citations"]) == 1
        assert body["citations"][0]["tmdb_id"] == 999

        # Verify the settings were passed down to the agent run
        assert len(captured_settings) == 1
        s = captured_settings[0]
        assert s.clarification_answer == "no"
        assert s.franchise_sibling_ids == [GLASS_ONION_TMDB_ID]

    def test_second_ask_with_include_answer_returns_normal_recs(self, monkeypatch) -> None:
        """'yes' answer also returns normal recs (includes path, no exclusion)."""
        def stub_run(query: str, settings: AgentSettings | None = None) -> AgentRunResult:
            return _NORMAL_RESULT

        with _fresh_client(stub_run, monkeypatch) as c:
            resp = c.post(
                "/ask",
                json={
                    "query": "films like Knives Out",
                    "clarification_answer": "yes",
                    "franchise_sibling_ids": [GLASS_ONION_TMDB_ID],
                },
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["needs_clarification"] is False
        assert len(body["citations"]) == 1

    def test_non_franchise_query_unchanged_response_shape(self, monkeypatch) -> None:
        """A non-franchise query returns the normal response without clarify fields set."""
        def stub_run(query: str, settings: AgentSettings | None = None) -> AgentRunResult:
            return _NORMAL_RESULT

        with _fresh_client(stub_run, monkeypatch) as c:
            resp = c.post("/ask", json={"query": "recommend something slow"})

        assert resp.status_code == 200
        body = resp.json()
        assert body["needs_clarification"] is False
        assert body["clarification_question"] is None
        assert body["franchise_sibling_ids"] == []
        assert len(body["citations"]) == 1

    def test_ask_request_clarification_fields_optional_backward_compat(self, monkeypatch) -> None:
        """Existing clients not sending clarification fields still work (AC-3 regression)."""
        def stub_run(query: str, settings: AgentSettings | None = None) -> AgentRunResult:
            return _NORMAL_RESULT

        with _fresh_client(stub_run, monkeypatch) as c:
            # Old-style request without any franchise fields
            resp = c.post("/ask", json={"query": "slow meditative film"})

        assert resp.status_code == 200
        body = resp.json()
        # New fields default gracefully
        assert "needs_clarification" in body
        assert "franchise_sibling_ids" in body
