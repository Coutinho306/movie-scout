"""Singleton QdrantClient keyed on (url, api_key)."""

from __future__ import annotations

import os

from qdrant_client import QdrantClient

_registry: dict[tuple[str, str], QdrantClient] = {}


def get_qdrant_client(
    url: str | None = None,
    api_key: str | None = None,
) -> QdrantClient:
    """Return a cached QdrantClient for (url, api_key).

    Falls back to QDRANT_URL / QDRANT_API_KEY env vars when not provided.
    """
    resolved_url = url or os.environ["QDRANT_URL"]
    resolved_key = api_key or os.environ.get("QDRANT_API_KEY", "")
    key = (resolved_url, resolved_key)
    if key not in _registry:
        _registry[key] = QdrantClient(url=resolved_url, api_key=resolved_key or None)
    return _registry[key]
