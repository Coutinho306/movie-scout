"""AC-9: Two distinct profiles yield different taste orderings for the same query.

Both orderings differ from cold start (no profile).

This test is deterministic (no live LLM/Qdrant calls needed) — it directly
exercises the match_taste_tool + score_against_taste_with_vectors path with
synthetic profiles and hits.
"""

from __future__ import annotations

import math

from ingestion.models import TasteProfile
from retrieval.models import MovieHit
from agent.tools.taste_matcher import match_taste_tool


def _norm(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


def _profile_from_centroid(centroid: list[float]) -> TasteProfile:
    return TasteProfile(
        centroid=centroid,
        film_count=3,
        rated_count=3,
        liked_count=0,
        top_genre_ids=[28],
        genre_weights={"Action": 1.0},
        created_at="2026-07-09T00:00:00+00:00",
    )


def _make_hit(
    tmdb_id: int, score: float, vector: list[float]
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


def test_two_profiles_yield_different_orderings() -> None:
    """AC-9: Two distinct profiles produce different re-rankings for identical hits.

    Setup: Film A has the highest retrieval score (cold-start order: A, B, C).
    Profile 1 aligns with Film B → B promoted to top.
    Profile 2 aligns with Film C → C promoted to top.
    Cold start preserves [A, B, C].
    """
    # Three orthogonal unit vectors
    direction_a = _norm([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    direction_b = _norm([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    direction_c = _norm([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # Descending retrieval scores so cold-start order is A > B > C
    # alpha=0.5: blended = 0.5*score + 0.5*taste_score
    # For profile aligned with B: blended_B = 0.5*0.6 + 0.5*1.0 = 0.8
    #                             blended_A = 0.5*0.9 + 0.5*0.0 = 0.45
    # → B ranks first with profile_1 but NOT without
    film_a = _make_hit(1, 0.9, direction_a)  # highest retrieval, tastes like A
    film_b = _make_hit(2, 0.6, direction_b)  # mid retrieval, tastes like B
    film_c = _make_hit(3, 0.3, direction_c)  # lowest retrieval, tastes like C

    hits = [film_a, film_b, film_c]

    # Profile 1: aligned with direction_b → Film B promoted to first
    profile_1 = _profile_from_centroid(direction_b)
    # Profile 2: aligned with direction_c → Film C promoted to first
    profile_2 = _profile_from_centroid(direction_c)

    result_1 = match_taste_tool(list(hits), profile=profile_1)
    result_2 = match_taste_tool(list(hits), profile=profile_2)
    result_cold = match_taste_tool(list(hits), profile=None)

    order_1 = [h.tmdb_id for h in result_1]
    order_2 = [h.tmdb_id for h in result_2]
    order_cold = [h.tmdb_id for h in result_cold]

    # Cold start preserves original retrieval order (A > B > C)
    assert order_cold == [1, 2, 3], (
        f"Cold start should preserve retrieval order [1,2,3], got {order_cold}"
    )

    # Profile 1 (B-aligned) promotes Film B to first
    assert order_1[0] == 2, f"Profile 1 should rank Film B first, got {order_1}"
    # Profile 2 (C-aligned) promotes Film C to first
    assert order_2[0] == 3, f"Profile 2 should rank Film C first, got {order_2}"

    # The two profiles produce different orderings
    assert order_1 != order_2, (
        f"Profiles 1 and 2 produced identical orderings: {order_1}"
    )

    # Both differ from cold start
    assert order_1 != order_cold, "Profile 1 ordering should differ from cold start"
    assert order_2 != order_cold, "Profile 2 ordering should differ from cold start"


def test_taste_scores_differ_between_profiles() -> None:
    """taste_score values are different for different profiles on the same hit."""
    dim = 8
    vec = _norm([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    anti = [-x for x in vec]

    profile_aligned = _profile_from_centroid(vec)
    profile_anti = _profile_from_centroid(anti)

    hit = _make_hit(99, 0.5, vec)

    result_aligned = match_taste_tool([hit], profile=profile_aligned)
    result_anti = match_taste_tool([hit], profile=profile_anti)

    # Aligned: taste_score ≈ 1.0; anti-aligned: taste_score ≈ 0.0 (clamped)
    assert result_aligned[0].taste_score > 0.99
    assert result_anti[0].taste_score < 0.01
