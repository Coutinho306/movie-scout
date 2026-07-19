"""Integration tests: search_movies rerank=True wires cross_encode_rerank.

These tests mock Qdrant and the embedder so they run offline. The goal is to
verify that:

1. rerank=True fetches min(k*3, 30) candidates, calls cross_encode_rerank over
   the widened pool, and truncates to k.
2. rerank=False fetches exactly k candidates and never imports/calls the
   reranker — the result order is the raw Qdrant order, unchanged.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest
from qdrant_client.models import ScoredPoint

from retrieval.config import RetrievalSettings
from retrieval.models import MovieHit
from retrieval.movies import search_movies


def _make_scored_point(tmdb_id: int, score: float) -> ScoredPoint:
    """Build a minimal ScoredPoint carrying only the payload fields search_movies uses."""
    pt = MagicMock(spec=ScoredPoint)
    pt.payload = {
        "tmdb_id": tmdb_id,
        "title": f"Movie {tmdb_id}",
        "year": 2000 + tmdb_id,
        "overview": f"Overview of movie {tmdb_id}.",
        "genres": ["Drama"],
        "vote_average": 7.0,
    }
    pt.score = score
    pt.vector = [0.1] * 4  # dummy vector, dimension irrelevant for these tests
    return pt


def _make_mock_qdrant(points: list[ScoredPoint]) -> MagicMock:
    client = MagicMock()
    result = MagicMock()
    result.points = points
    client.query_points.return_value = result
    return client


def _make_mock_embedder() -> MagicMock:
    embedder = MagicMock()
    embedder.embed_single.return_value = [0.1, 0.2, 0.3, 0.4]
    return embedder


# ---------------------------------------------------------------------------
# rerank=False: fetch exactly k, no reranker import or call
# ---------------------------------------------------------------------------

def test_rerank_off_fetches_exactly_k() -> None:
    """When rerank=False, query_points is called with limit=k."""
    k = 5
    points = [_make_scored_point(i, 1.0 - i * 0.1) for i in range(k)]
    mock_client = _make_mock_qdrant(points)
    settings = RetrievalSettings(rerank=False)

    with (
        patch("retrieval.movies.get_qdrant_client", return_value=mock_client),
        patch("retrieval.movies.get_embedder", return_value=_make_mock_embedder()),
        patch("retrieval.movies._build_filter", return_value=None),
        patch("retrieval.rerank.cross_encode_rerank") as mock_rerank,
    ):
        hits = search_movies("quiet drama", settings=settings, k=k)

    # Qdrant called with limit=k (no widening)
    [qp_call] = mock_client.query_points.call_args_list
    assert qp_call.kwargs["limit"] == k, (
        f"Expected fetch limit={k} when rerank=False, got {qp_call.kwargs['limit']}"
    )
    # cross_encode_rerank never called
    mock_rerank.assert_not_called()
    # Returned exactly k hits in raw Qdrant order
    assert len(hits) == k
    assert [h.tmdb_id for h in hits] == list(range(k))


# ---------------------------------------------------------------------------
# rerank=True: fetch min(k*3, 30) candidates, rerank, truncate to k
# ---------------------------------------------------------------------------

def test_rerank_on_fetches_widened_pool_and_truncates() -> None:
    """When rerank=True, query_points is called with limit=min(k*3, 30) and
    result is truncated to k after reranking."""
    k = 5
    fetch_k = min(k * 3, 30)  # 15
    # Simulate Qdrant returning the full widened pool
    points = [_make_scored_point(i, 1.0 - i * 0.05) for i in range(fetch_k)]
    mock_client = _make_mock_qdrant(points)
    settings = RetrievalSettings(rerank=True)

    # cross_encode_rerank reverses the list (a stable, detectable reordering)
    def _reverse_hits(query: str, hits: list[MovieHit], **kwargs: Any) -> list[MovieHit]:
        return list(reversed(hits))

    # cross_encode_rerank is a lazy import inside search_movies; patch at source.
    with (
        patch("retrieval.movies.get_qdrant_client", return_value=mock_client),
        patch("retrieval.movies.get_embedder", return_value=_make_mock_embedder()),
        patch("retrieval.movies._build_filter", return_value=None),
        patch("retrieval.rerank.cross_encode_rerank", side_effect=_reverse_hits),
    ):
        hits = search_movies("quiet drama", settings=settings, k=k)

    # Qdrant called with the widened limit
    [qp_call] = mock_client.query_points.call_args_list
    assert qp_call.kwargs["limit"] == fetch_k, (
        f"Expected fetch limit={fetch_k} when rerank=True (k={k}), "
        f"got {qp_call.kwargs['limit']}"
    )
    # Result truncated to k
    assert len(hits) == k
    # Reversed order: tmdb_ids should be [fetch_k-1, fetch_k-2, ..., fetch_k-k]
    expected_ids = list(range(fetch_k - 1, fetch_k - 1 - k, -1))
    assert [h.tmdb_id for h in hits] == expected_ids, (
        f"Expected reversed ids {expected_ids}, got {[h.tmdb_id for h in hits]}"
    )


def test_rerank_produces_different_order_than_no_rerank() -> None:
    """Proves that enabling rerank changes the result order compared to off-path.

    Uses a carefully crafted pool where the cross-encoder would naturally
    prefer a different ordering than raw similarity scores — simulated here
    by a deterministic reverse reranker.
    """
    k = 3
    fetch_k = min(k * 3, 30)
    points_for_rerank = [_make_scored_point(i, 1.0 - i * 0.05) for i in range(fetch_k)]
    points_for_dense = [_make_scored_point(i, 1.0 - i * 0.05) for i in range(k)]

    def _reverse_hits(query: str, hits: list[MovieHit], **kwargs: Any) -> list[MovieHit]:
        return list(reversed(hits))

    # --- rerank=False ---
    mock_client_off = _make_mock_qdrant(points_for_dense)
    settings_off = RetrievalSettings(rerank=False)
    with (
        patch("retrieval.movies.get_qdrant_client", return_value=mock_client_off),
        patch("retrieval.movies.get_embedder", return_value=_make_mock_embedder()),
        patch("retrieval.movies._build_filter", return_value=None),
    ):
        hits_off = search_movies("quiet drama", settings=settings_off, k=k)

    # --- rerank=True ---
    mock_client_on = _make_mock_qdrant(points_for_rerank)
    settings_on = RetrievalSettings(rerank=True)
    with (
        patch("retrieval.movies.get_qdrant_client", return_value=mock_client_on),
        patch("retrieval.movies.get_embedder", return_value=_make_mock_embedder()),
        patch("retrieval.movies._build_filter", return_value=None),
        patch("retrieval.rerank.cross_encode_rerank", side_effect=_reverse_hits),
    ):
        hits_on = search_movies("quiet drama", settings=settings_on, k=k)

    ids_off = [h.tmdb_id for h in hits_off]
    ids_on = [h.tmdb_id for h in hits_on]

    assert ids_off != ids_on, (
        "rerank=True and rerank=False produced the same ordering — "
        "the reranker is not changing the result order"
    )
    assert len(hits_on) == k
