"""Tests for retrieval.movies.search_movies.

Integration tests (marked with the `settings` fixture) run against live Qdrant.
Unit tests for dense_score plumbing use fully-mocked Qdrant + embedder.
"""

import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from retrieval.config import RetrievalSettings
from retrieval.models import MovieFilters, MovieHit
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


# ---------------------------------------------------------------------------
# Unit tests: dense_score plumbing (no live Qdrant, no real embedder)
# ---------------------------------------------------------------------------


def _make_scored_point(
    tmdb_id: int,
    score: float,
    vector: list[float] | None,
) -> MagicMock:
    """Build a mock ScoredPoint with payload and optional vector."""
    point = MagicMock()
    point.score = score
    point.payload = {
        "tmdb_id": tmdb_id,
        "title": f"Film {tmdb_id}",
        "year": 2000,
        "overview": "A test film.",
        "genres": ["Drama"],
        "vote_average": 7.0,
    }
    point.vector = vector  # bare list (single unnamed vector)
    return point


def _unit_search(
    query_vec: list[float],
    points: list[MagicMock],
    *,
    hybrid: bool = False,
) -> tuple[list[MovieHit], int]:
    """Run search_movies with fully-mocked Qdrant client and embedder.

    Returns (hits, qdrant_call_count) so tests can assert no extra calls.
    """
    settings = RetrievalSettings(hybrid=hybrid)

    mock_embedder = MagicMock()
    mock_embedder.embed_single.return_value = query_vec

    mock_client = MagicMock()
    query_result = MagicMock()
    query_result.points = points
    mock_client.query_points.return_value = query_result

    if hybrid:
        # BM25 sparse embedding mock
        mock_sv = MagicMock()
        mock_sv.indices = np.array([0, 1])
        mock_sv.values = np.array([0.5, 0.5])
        mock_bm25 = MagicMock()
        mock_bm25.embed.return_value = iter([mock_sv])

        with (
            patch("retrieval.movies.get_embedder", return_value=mock_embedder),
            patch("retrieval.movies.get_qdrant_client", return_value=mock_client),
            patch("retrieval.movies._get_bm25_model", return_value=mock_bm25),
        ):
            hits = search_movies("test query", settings=settings, k=len(points))
    else:
        with (
            patch("retrieval.movies.get_embedder", return_value=mock_embedder),
            patch("retrieval.movies.get_qdrant_client", return_value=mock_client),
        ):
            hits = search_movies("test query", settings=settings, k=len(points))

    call_count = mock_client.query_points.call_count
    return hits, call_count


class TestDenseScorePlumbing:
    """Unit tests for the dense_score field — AC-2, AC-3."""

    def test_dense_score_equals_cosine_of_query_vec_vs_hit_vector(self) -> None:
        """dense_score must equal the exact cosine similarity of query_vec vs hit.vector."""
        query_vec = [1.0, 0.0, 0.0]
        hit_vec = [0.6, 0.8, 0.0]  # cos(query, hit) = 0.6 / (1 * 1) = 0.6
        point = _make_scored_point(101, 0.6, hit_vec)

        hits, _ = _unit_search(query_vec, [point], hybrid=False)

        assert len(hits) == 1
        expected_cos = float(
            np.dot(query_vec, hit_vec)
            / (np.linalg.norm(query_vec) * np.linalg.norm(hit_vec))
        )
        assert math.isclose(hits[0].dense_score, expected_cos, abs_tol=1e-6)

    def test_none_vector_hit_yields_dense_score_zero(self) -> None:
        """A hit with vector=None must get dense_score=0.0 (safe-low fallback)."""
        query_vec = [1.0, 0.0, 0.0]
        point = _make_scored_point(102, 0.5, None)

        hits, _ = _unit_search(query_vec, [point], hybrid=False)

        assert len(hits) == 1
        assert hits[0].dense_score == 0.0

    def test_dense_mode_dense_score_is_on_cosine_scale(self) -> None:
        """In dense mode dense_score must be in [-1, 1] (cosine range)."""
        query_vec = [0.7071, 0.7071, 0.0]
        hit_vec = [0.5774, 0.5774, 0.5774]
        point = _make_scored_point(103, 0.942, hit_vec)

        hits, _ = _unit_search(query_vec, [point], hybrid=False)

        assert len(hits) == 1
        assert -1.0 <= hits[0].dense_score <= 1.0

    def test_hybrid_path_issues_no_second_qdrant_call(self) -> None:
        """Hybrid path must not issue more than one query_points call (AC-2)."""
        query_vec = [1.0, 0.0, 0.0]
        hit_vec = [0.8, 0.6, 0.0]
        point = _make_scored_point(201, 1.0, hit_vec)  # RRF score=1.0 (rank-1)

        _, call_count = _unit_search(query_vec, [point], hybrid=True)

        assert call_count == 1, (
            f"Expected exactly 1 query_points call in hybrid path, got {call_count}"
        )

    def test_hybrid_hit_dense_score_computed_correctly(self) -> None:
        """Hybrid hit gets the same cosine dense_score as dense mode for same vectors."""
        query_vec = [1.0, 0.0, 0.0]
        hit_vec = [0.6, 0.8, 0.0]
        # In hybrid mode score is an RRF fraction; dense_score must still be the cosine
        point = _make_scored_point(202, 1.0, hit_vec)  # RRF rank-1 score

        hits, _ = _unit_search(query_vec, [point], hybrid=True)

        assert len(hits) == 1
        expected_cos = float(
            np.dot(query_vec, hit_vec)
            / (np.linalg.norm(query_vec) * np.linalg.norm(hit_vec))
        )
        assert math.isclose(hits[0].dense_score, expected_cos, abs_tol=1e-6)

    def test_dense_score_survives_model_dump(self) -> None:
        """dense_score must appear in model_dump() output (not excluded like vector)."""
        query_vec = [1.0, 0.0, 0.0]
        hit_vec = [0.6, 0.8, 0.0]
        point = _make_scored_point(301, 0.6, hit_vec)

        hits, _ = _unit_search(query_vec, [point], hybrid=False)

        assert len(hits) == 1
        dumped = hits[0].model_dump()
        assert "dense_score" in dumped, "dense_score must survive model_dump()"
        assert "vector" not in dumped, "vector must remain excluded from model_dump()"
        assert isinstance(dumped["dense_score"], float)
