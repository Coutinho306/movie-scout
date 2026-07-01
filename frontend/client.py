"""HTTP client for the Movie Scout FastAPI backend.

All backend calls funnel through here — the Streamlit module never touches
httpx directly. Sync client: Streamlit reruns top-to-bottom per interaction,
so async buys nothing.
"""

from __future__ import annotations

import os

import httpx

DEFAULT_TIMEOUT = 60.0  # agent runs can take 20–40s with web fallback


def _base_url() -> str:
    return os.environ.get("API_BASE_URL", "http://localhost:8000").rstrip("/")


def _client(transport: httpx.BaseTransport | None = None) -> httpx.Client:
    return httpx.Client(
        base_url=_base_url(),
        timeout=DEFAULT_TIMEOUT,
        transport=transport,
    )


def ask(query: str, session_id: str | None = None, *, transport=None) -> dict:
    """POST /ask — return the AskResponse-shaped JSON dict. Raises on HTTP error."""
    with _client(transport) as client:
        resp = client.post("/ask", json={"query": query, "session_id": session_id})
        resp.raise_for_status()
        return resp.json()


def feedback(
    run_id: str, rating: str, comment: str | None = None, *, transport=None
) -> None:
    """POST /feedback — fire a thumbs up/down. Raises on HTTP error."""
    with _client(transport) as client:
        resp = client.post(
            "/feedback",
            json={"run_id": run_id, "rating": rating, "comment": comment},
        )
        resp.raise_for_status()
