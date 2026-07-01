"""frontend.client tests — httpx.MockTransport, no live backend."""

from __future__ import annotations

from uuid import uuid4

import httpx
import pytest

from frontend import client

RUN_ID = str(uuid4())


def _ask_transport() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/ask"
        assert request.method == "POST"
        return httpx.Response(
            200,
            json={
                "run_id": RUN_ID,
                "final_answer": "Watch Paris, Texas.",
                "citations": [
                    {
                        "tmdb_id": 655,
                        "title": "Paris, Texas",
                        "year": 1984,
                        "why_for_you": "Slow and meditative.",
                        "provider_hint": None,
                    }
                ],
                "latency_ms": 123.4,
                "cost_usd": 0.0012,
                "tool_calls": 2,
            },
        )

    return httpx.MockTransport(handler)


def test_ask_returns_response_dict() -> None:
    resp = client.ask("slow film", transport=_ask_transport())
    assert resp["run_id"] == RUN_ID
    assert resp["final_answer"] == "Watch Paris, Texas."
    assert resp["citations"][0]["tmdb_id"] == 655
    assert resp["tool_calls"] == 2


def test_ask_raises_on_error() -> None:
    transport = httpx.MockTransport(lambda req: httpx.Response(500))
    with pytest.raises(httpx.HTTPStatusError):
        client.ask("boom", transport=transport)


def test_feedback_posts_payload() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/feedback"
        import json

        captured.update(json.loads(request.content))
        return httpx.Response(204)

    client.feedback(RUN_ID, "up", "great", transport=httpx.MockTransport(handler))
    assert captured == {"run_id": RUN_ID, "rating": "up", "comment": "great"}


def test_feedback_raises_on_error() -> None:
    transport = httpx.MockTransport(lambda req: httpx.Response(500))
    with pytest.raises(httpx.HTTPStatusError):
        client.feedback(RUN_ID, "down", transport=transport)
