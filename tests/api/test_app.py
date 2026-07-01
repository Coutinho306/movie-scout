"""API route tests — agent + pool faked via dependency overrides, no live calls."""

from __future__ import annotations

from uuid import uuid4

from starlette.testclient import TestClient


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
