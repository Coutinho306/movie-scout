"""FastAPI TestClient with the agent + pool dependencies overridden — no live calls."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from agent.config import AgentSettings
from agent.state import AgentRunResult, RecItem
from api.dependencies import get_agent_run_fn, get_agent_settings, get_pg_pool
from api.fastapi_app import create_app

STUB_RESULT = AgentRunResult(
    final_answer="A slow, meditative film: Paris, Texas (1984).",
    citations=[
        RecItem(
            tmdb_id=655,
            title="Paris, Texas",
            year=1984,
            why_for_you="Meditative, wandering, quiet.",
        )
    ],
    tool_calls=2,
    latency_ms=123.4,
    cost_usd=0.0012,
    orchestrator_turns=1,
    rag_calls=2,
    web_calls=0,
)


def _stub_agent_run(query: str, settings: AgentSettings | None = None) -> AgentRunResult:
    return STUB_RESULT


@pytest.fixture
def client() -> TestClient:
    app = create_app()
    app.dependency_overrides[get_agent_run_fn] = lambda: _stub_agent_run
    app.dependency_overrides[get_agent_settings] = lambda: AgentSettings()
    app.dependency_overrides[get_pg_pool] = lambda: None  # persistence disabled
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
