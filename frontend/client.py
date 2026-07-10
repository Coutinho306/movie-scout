"""HTTP client for the Movie Scout FastAPI backend.

All backend calls funnel through here — the Streamlit module never touches
httpx directly. Sync client: Streamlit reruns top-to-bottom per interaction,
so async buys nothing.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

DEFAULT_TIMEOUT = 90.0  # taste-profile upload can be slow (TMDB resolution)
ASK_TIMEOUT = 60.0     # agent runs can take 20–40s with web fallback


def _base_url() -> str:
    return os.environ.get("API_BASE_URL", "http://localhost:8000").rstrip("/")


def _client(
    transport: httpx.BaseTransport | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> httpx.Client:
    return httpx.Client(
        base_url=_base_url(),
        timeout=timeout,
        transport=transport,
    )


def ask(
    query: str,
    session_id: str | None = None,
    *,
    taste_profile: dict | None = None,
    clarification_answer: str | None = None,
    franchise_sibling_ids: list[int] | None = None,
    transport: httpx.BaseTransport | None = None,
) -> dict:
    """POST /ask — return the AskResponse-shaped JSON dict. Raises on HTTP error.

    ``taste_profile`` is the dict from a prior ``upload_taste`` call (the
    ``profile`` field). When absent, the server uses cold-start (no taste).

    ``clarification_answer`` and ``franchise_sibling_ids`` are set on the
    second call after a ``needs_clarification=True`` response — they carry the
    stateless franchise round-trip (AC-4, AC-9).
    """
    body: dict[str, Any] = {"query": query, "session_id": session_id}
    if taste_profile is not None:
        body["taste_profile"] = taste_profile
    if clarification_answer is not None:
        body["clarification_answer"] = clarification_answer
    if franchise_sibling_ids:
        body["franchise_sibling_ids"] = franchise_sibling_ids
    with _client(transport, timeout=ASK_TIMEOUT) as client:
        resp = client.post("/ask", json=body)
        resp.raise_for_status()
        return resp.json()


def upload_taste(
    file_bytes: bytes,
    filename: str = "ratings.csv",
    *,
    transport: httpx.BaseTransport | None = None,
) -> dict:
    """POST /taste-profile — upload a Letterboxd CSV or ZIP and return TasteProfileResponse JSON.

    Returns a dict with keys: profile, resolved, tmdb_miss, out_of_corpus, total_input.
    Raises httpx.HTTPStatusError on HTTP error.
    """
    content_type = "application/zip" if filename.lower().endswith(".zip") else "text/csv"
    with _client(transport) as client:
        resp = client.post(
            "/taste-profile",
            files={"file": (filename, file_bytes, content_type)},
        )
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
