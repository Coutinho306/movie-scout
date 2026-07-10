"""AC-8: /ask with no profile → valid response, retrieval order, no FileNotFoundError."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import api.fastapi_app as fapp
from agent.config import AgentSettings
from agent.state import AgentRunResult, RecItem
from api.dependencies import get_agent_run_fn, get_agent_settings, get_pg_pool
from api.fastapi_app import create_app
from slowapi import Limiter

STUB_RESULT = AgentRunResult(
    final_answer="A slow film: Jeanne Dielman (1975).",
    citations=[
        RecItem(
            tmdb_id=42,
            title="Jeanne Dielman",
            year=1975,
            why_for_you="Slow, meditative, dense.",
        )
    ],
    tool_calls=1,
    latency_ms=50.0,
    cost_usd=0.0005,
    orchestrator_turns=1,
    rag_calls=1,
    web_calls=0,
)


def _stub_run(query: str, settings: AgentSettings | None = None) -> AgentRunResult:
    # Cold start: profile on settings must be None
    assert settings is not None
    assert settings.taste_profile is None, (
        f"Expected cold start (profile=None) but got: {settings.taste_profile}"
    )
    return STUB_RESULT


def test_ask_no_profile_cold_start(monkeypatch: pytest.MonkeyPatch) -> None:
    """/ask with no taste_profile → cold start (settings.taste_profile is None)."""
    monkeypatch.setattr(fapp, "limiter", Limiter(key_func=fapp._client_key))

    app = create_app()
    app.dependency_overrides[get_agent_run_fn] = lambda: _stub_run
    app.dependency_overrides[get_agent_settings] = lambda: AgentSettings()
    app.dependency_overrides[get_pg_pool] = lambda: None

    with TestClient(app) as c:
        resp = c.post("/ask", json={"query": "slow meditative film"})

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["final_answer"].startswith("A slow film")
    app.dependency_overrides.clear()


def test_ask_with_profile_threads_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    """/ask with taste_profile in body threads it onto per-request AgentSettings."""
    from ingestion.models import TasteProfile

    sample_profile = TasteProfile(
        centroid=[0.1] * 1536,
        film_count=5,
        rated_count=5,
        liked_count=0,
        top_genre_ids=[28],
        genre_weights={"Action": 1.0},
        created_at="2026-07-09T00:00:00+00:00",
    )

    received_profile: list[TasteProfile | None] = []

    def _stub_with_profile(
        query: str, settings: AgentSettings | None = None
    ) -> AgentRunResult:
        received_profile.append(settings.taste_profile if settings else None)
        return STUB_RESULT

    monkeypatch.setattr(fapp, "limiter", Limiter(key_func=fapp._client_key))

    app = create_app()
    app.dependency_overrides[get_agent_run_fn] = lambda: _stub_with_profile
    app.dependency_overrides[get_agent_settings] = lambda: AgentSettings()
    app.dependency_overrides[get_pg_pool] = lambda: None

    with TestClient(app) as c:
        resp = c.post(
            "/ask",
            json={
                "query": "slow film",
                "taste_profile": sample_profile.model_dump(),
            },
        )

    assert resp.status_code == 200, resp.text
    assert len(received_profile) == 1
    p = received_profile[0]
    assert p is not None
    assert p.film_count == 5
    assert p.centroid[:3] == [0.1, 0.1, 0.1]
    app.dependency_overrides.clear()
