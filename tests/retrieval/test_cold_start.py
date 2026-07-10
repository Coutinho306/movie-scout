"""Tests for cold-start behavior — AC-8.

A full /ask with no profile returns a valid response in retrieval order
with no FileNotFoundError and taste ignored.
"""

from __future__ import annotations

from retrieval.models import MovieHit
from retrieval.taste import score_against_taste, score_against_taste_with_vectors
from agent.tools.taste_matcher import match_taste_tool


def _make_hit(
    tmdb_id: int, score: float, vector: list[float] | None = None
) -> MovieHit:
    return MovieHit(
        tmdb_id=tmdb_id,
        title=f"Film {tmdb_id}",
        year=2000,
        overview="",
        genres=[],
        vote_average=7.0,
        score=score,
        vector=vector,
    )


def test_match_taste_tool_cold_start_returns_retrieval_order() -> None:
    """profile=None → cold start: hits returned in original retrieval order, no error."""
    hits = [
        _make_hit(1, 0.9),
        _make_hit(2, 0.5),
        _make_hit(3, 0.7),
    ]
    # Cold start: profile=None
    result = match_taste_tool(hits, profile=None)

    assert len(result) == len(hits)
    # Must preserve original retrieval order (not re-sorted by taste)
    assert [h.tmdb_id for h in result] == [1, 2, 3]


def test_match_taste_tool_cold_start_no_file_error(tmp_path) -> None:
    """Cold start must not raise FileNotFoundError even when taste_profile.json is absent."""
    import os

    # Ensure we're not looking at any data/ directory with taste_profile.json
    hits = [_make_hit(10, 0.8), _make_hit(20, 0.6)]

    # This must not raise FileNotFoundError
    result = match_taste_tool(hits, profile=None)
    assert len(result) == 2


def test_match_taste_tool_cold_start_empty_hits() -> None:
    """Cold start with empty hits is not an error."""
    result = match_taste_tool([], profile=None)
    assert result == []


def test_cold_start_vs_warm_start_ordering() -> None:
    """Cold start (profile=None) preserves retrieval order;
    warm start (profile provided) can reorder by taste."""
    from ingestion.models import TasteProfile
    import math

    dim = 8

    def _norm(v: list[float]) -> list[float]:
        n = math.sqrt(sum(x * x for x in v))
        return [x / n for x in v]

    centroid = _norm([1.0] * dim)
    anti = [-c for c in centroid]

    # Hit 1: higher retrieval score, but anti-taste vector
    # Hit 2: lower retrieval score, on-taste vector
    hit1 = _make_hit(1, 0.9, vector=anti)
    hit2 = _make_hit(2, 0.4, vector=centroid[:])

    profile = TasteProfile(
        centroid=centroid,
        film_count=2,
        rated_count=2,
        liked_count=0,
        top_genre_ids=[28],
        genre_weights={"Action": 1.0},
        created_at="2026-07-09T00:00:00+00:00",
    )

    # Cold start: retrieval order preserved (hit1 first)
    cold_result = match_taste_tool([hit1, hit2], profile=None)
    assert cold_result[0].tmdb_id == 1

    # Warm start: taste reorders (hit2 first — on-taste)
    warm_result = match_taste_tool([hit1, hit2], profile=profile)
    assert warm_result[0].tmdb_id == 2
