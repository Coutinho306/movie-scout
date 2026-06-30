"""Integration tests for retrieval.reviews.search_reviews."""

import pytest

from retrieval.config import RetrievalSettings
from retrieval.reviews import search_reviews


def test_search_reviews_returns_hits(settings: RetrievalSettings) -> None:
    hits = search_reviews("dark atmospheric slow burn", settings=settings, k=5)
    assert len(hits) > 0
    assert all(h.tmdb_id > 0 for h in hits)
    assert all(h.chunk_text for h in hits)


def test_search_reviews_tmdb_id_filter(settings: RetrievalSettings) -> None:
    # First get some tmdb_ids from an unrestricted search
    hits = search_reviews("cinematic masterpiece", settings=settings, k=20)
    if not hits:
        pytest.skip("empty collection")
    first_id = hits[0].tmdb_id
    filtered = search_reviews(
        "cinematic masterpiece", settings=settings, k=10, tmdb_ids=[first_id]
    )
    if filtered:
        assert all(h.tmdb_id == first_id for h in filtered)


def test_search_reviews_hybrid_fallback(hybrid_settings: RetrievalSettings) -> None:
    hits = search_reviews("beautiful cinematography", settings=hybrid_settings, k=5)
    assert isinstance(hits, list)
