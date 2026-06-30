"""Tool: TMDB streaming provider lookup (watch_region=BR default)."""

from __future__ import annotations

import logging
import os

import requests

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.themoviedb.org/3"


def _auth_headers() -> dict[str, str]:
    token = os.getenv("TMDB_API_KEY", "")
    return {"Authorization": f"Bearer {token}", "accept": "application/json"}


def get_providers(tmdb_id: int, region: str = "BR") -> list[str]:
    """Return flatrate provider names for ``tmdb_id`` in ``region`` (empty on failure).

    One retry on network error before giving up.
    """
    url = f"{_BASE_URL}/movie/{tmdb_id}/watch/providers"
    headers = _auth_headers()

    last_exc: Exception | None = None
    for attempt in range(2):
        try:
            resp = requests.get(url, headers=headers, timeout=10.0)
            resp.raise_for_status()
            data = resp.json()
            region_data = data.get("results", {}).get(region, {})
            flatrate = region_data.get("flatrate", [])
            return [p["provider_name"] for p in flatrate if "provider_name" in p]
        except requests.HTTPError as exc:
            # A 4xx (e.g. unknown id) won't fix on retry — bail immediately.
            logger.warning("TMDB providers HTTP error for %s: %s", tmdb_id, exc)
            return []
        except requests.RequestException as exc:
            last_exc = exc
            if attempt == 0:
                continue
    logger.warning("TMDB providers lookup failed for %s: %s", tmdb_id, last_exc)
    return []
