"""Cached TMDB poster-URL lookup. Opportunistic — returns None on any failure."""

from __future__ import annotations

import os

import httpx
import streamlit as st

TMDB_BASE = "https://api.themoviedb.org/3"
IMAGE_BASE = "https://image.tmdb.org/t/p/w500"


@st.cache_data(ttl=86400)
def poster_url(tmdb_id: int) -> str | None:
    """Return a w500 poster URL for the movie, or None if unavailable.

    Cached 24h. Missing TMDB key, 404, or timeout all yield None — the UI
    renders the card without an image.
    """
    api_key = os.environ.get("TMDB_API_KEY")
    if not api_key:
        return None
    try:
        resp = httpx.get(
            f"{TMDB_BASE}/movie/{tmdb_id}",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10,
        )
        if resp.status_code != 200:
            return None
        path = resp.json().get("poster_path")
        return f"{IMAGE_BASE}{path}" if path else None
    except httpx.HTTPError:
        return None
