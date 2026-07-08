"""Integration tests for retrieval.reviews.search_reviews (dense-only).

tmdb_reviews has no sparse text field; search_reviews is unconditionally
dense-only (hybrid branch deleted in specs/0009-bm25-intent-routed-hybrid).
"""

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


def test_search_reviews_dense_only(settings: RetrievalSettings) -> None:
    """search_reviews is dense-only: settings.hybrid is ignored and no hybrid
    branch exists. Calling with hybrid=True returns results (not an error)
    and uses dense retrieval.
    """
    hybrid_settings = RetrievalSettings(hybrid=True)
    dense_hits = search_reviews("beautiful cinematography", settings=settings, k=5)
    # hybrid=True should give the same result as dense (no sparse field in reviews)
    hybrid_hits = search_reviews("beautiful cinematography", settings=hybrid_settings, k=5)
    assert isinstance(hybrid_hits, list)
    # Both calls go through the same dense path
    dense_ids = [h.tmdb_id for h in dense_hits]
    hybrid_ids = [h.tmdb_id for h in hybrid_hits]
    assert dense_ids == hybrid_ids, (
        "Expected identical results since reviews is always dense-only"
    )
