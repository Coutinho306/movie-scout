"""API route tests — agent + pool faked via dependency overrides, no live calls."""

from __future__ import annotations

from uuid import uuid4

from fastapi.testclient import TestClient

from agent.config import AgentSettings
from api.config import ApiSettings
from api.dependencies import get_agent_run_fn, get_agent_settings, get_pg_pool
import api.fastapi_app as fastapi_app
from api.fastapi_app import create_app


def test_ask_happy_path(client: TestClient) -> None:
    resp = client.post("/ask", json={"query": "slow meditative film"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["final_answer"].startswith("A slow, meditative film")
    assert body["tool_calls"] == 2
    assert body["cost_usd"] == 0.0012
    assert len(body["citations"]) == 1
    assert body["citations"][0]["tmdb_id"] == 655
    # run_id is a server-generated UUID
    assert len(body["run_id"]) == 36


def test_ask_validation_error(client: TestClient) -> None:
    resp = client.post("/ask", json={"not_query": "oops"})
    assert resp.status_code == 422


def test_feedback_204(client: TestClient) -> None:
    resp = client.post(
        "/feedback",
        json={"run_id": str(uuid4()), "rating": "up", "comment": "great"},
    )
    assert resp.status_code == 204


def test_feedback_validation_error(client: TestClient) -> None:
    resp = client.post(
        "/feedback",
        json={"run_id": str(uuid4()), "rating": "sideways"},
    )
    assert resp.status_code == 422


def test_healthz_200(client: TestClient) -> None:
    resp = client.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "qdrant" in body
    assert "openai" in body


def test_ask_rate_limited(monkeypatch) -> None:
    """/ask returns 429 once the per-client limit is exceeded."""
    from slowapi import Limiter

    from tests.api.conftest import _stub_agent_run

    # Swap in a fresh Limiter with its own storage so /ask limits registered by
    # other tests' apps on the shared module limiter can't leak into this one.
    monkeypatch.setattr(
        fastapi_app, "limiter", Limiter(key_func=fastapi_app._client_key)
    )
    app = create_app(ApiSettings(rate_limit="3/minute"))
    app.dependency_overrides[get_agent_run_fn] = lambda: _stub_agent_run
    app.dependency_overrides[get_agent_settings] = lambda: AgentSettings()
    app.dependency_overrides[get_pg_pool] = lambda: None
    with TestClient(app) as c:
        for _ in range(3):
            r = c.post("/ask", json={"query": "film"})
            assert r.status_code == 200
        assert c.post("/ask", json={"query": "film"}).status_code == 429
    app.dependency_overrides.clear()
