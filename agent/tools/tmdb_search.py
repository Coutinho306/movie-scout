"""Tool: TMDB movie search by title/year."""

from __future__ import annotations

import logging
import os

import requests

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.themoviedb.org/3"


def _auth_headers() -> dict[str, str]:
    token = os.getenv("TMDB_API_KEY", "")
    return {"Authorization": f"Bearer {token}", "accept": "application/json"}


def search_tmdb(title: str, year: int | None = None) -> int | None:
    """Return the tmdb_id of the first search result for ``title`` (or None)."""
    params: dict[str, str | int] = {"query": title}
    if year is not None:
        params["year"] = year

    try:
        resp = requests.get(
            f"{_BASE_URL}/search/movie",
            headers=_auth_headers(),
            params=params,
            timeout=10.0,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
    except requests.RequestException as exc:
        logger.warning("TMDB search failed for %r: %s", title, exc)
        return None

    if not results:
        return None
    return results[0].get("id")
