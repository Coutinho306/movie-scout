"""Integration tests for retrieval.movies.search_movies."""

import pytest

from retrieval.config import RetrievalSettings
from retrieval.models import MovieFilters
from retrieval.movies import search_movies


def test_search_movies_returns_hits(settings: RetrievalSettings) -> None:
    hits = search_movies("slow meditative contemplative film", settings=settings, k=5)
    assert len(hits) > 0
    assert all(h.tmdb_id > 0 for h in hits)
    assert all(0.0 <= h.score <= 1.1 for h in hits)


def test_search_movies_year_filter(settings: RetrievalSettings) -> None:
    hits = search_movies(
        "science fiction",
        settings=settings,
        k=10,
        filters=MovieFilters(year_min=2000, year_max=2015),
    )
    if hits:
        assert all(2000 <= h.year <= 2015 for h in hits)


def test_search_movies_genre_filter(settings: RetrievalSettings) -> None:
    hits = search_movies(
        "thriller suspense",
        settings=settings,
        k=10,
        filters=MovieFilters(genres=["Thriller", "Crime"]),
    )
    if hits:
        assert any(
            any(g in ["Thriller", "Crime"] for g in h.genres) for h in hits
        )


def test_search_movies_exclude_ids(settings: RetrievalSettings) -> None:
    hits_full = search_movies("drama", settings=settings, k=5)
    if not hits_full:
        pytest.skip("empty collection")
    exclude = {hits_full[0].tmdb_id}
    hits_excluded = search_movies(
        "drama", settings=settings, k=5, filters=MovieFilters(exclude_tmdb_ids=exclude)
    )
    ids = {h.tmdb_id for h in hits_excluded}
    assert exclude.isdisjoint(ids)


def test_search_movies_hybrid_rrf_fires(hybrid_settings: RetrievalSettings) -> None:
    """Hybrid RRF path fires and returns results that differ from dense for the
    same query (proves FusionQuery/Prefetch path is active, not a silent fallback).
    Uses a genre/mood query that benefits most from BM25 keyword overlap.
    """
    query = "a Action, Crime, Thriller film — Be careful who you trust."
    dense_settings = RetrievalSettings(hybrid=False)
    dense_hits = search_movies(query, settings=dense_settings, k=10)
    hybrid_hits = search_movies(query, settings=hybrid_settings, k=10)

    # Both paths must return results from the production collection
    assert len(dense_hits) > 0, "dense search returned no hits"
    assert len(hybrid_hits) > 0, "hybrid search returned no hits"

    # Hybrid must not be a byte-identical copy of dense — RRF reranks
    dense_ids = [h.tmdb_id for h in dense_hits]
    hybrid_ids = [h.tmdb_id for h in hybrid_hits]
    assert dense_ids != hybrid_ids, (
        "hybrid and dense returned identical result order — "
        "RRF is not firing (possible silent fallback to dense)"
    )
