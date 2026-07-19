"""Tests for retrieval.taste.score_against_taste."""

from pathlib import Path


from ingestion.scripts.compute_taste import load_taste_profile
from retrieval.models import MovieHit
from retrieval.taste import score_against_taste, score_against_taste_with_vectors


def _make_hit(tmdb_id: int, score: float, vector: list[float] | None = None) -> MovieHit:
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


def test_no_vector_fallback_sorts_by_retrieval_score() -> None:
    # Fallback path: no vectors → taste ignored, order is scaled retrieval order.
    profile = load_taste_profile(Path("data/taste_profile.json"))
    hits = [_make_hit(1, 0.9), _make_hit(2, 0.5), _make_hit(3, 0.7)]
    result = score_against_taste(hits, profile=profile)
    assert len(result) == 3
    assert all(h.taste_score == 0.0 for h in result)
    scores = [h.blended_score for h in result]
    assert scores == sorted(scores, reverse=True)


def test_taste_reorders_by_centroid() -> None:
    # The real taste path must promote the on-taste hit even when its retrieval
    # score is lower. This is the test that would have caught the no-op bug.
    profile = load_taste_profile(Path("data/taste_profile.json"))
    centroid = profile.centroid
    off_taste = [-c for c in centroid]  # anti-aligned → low cosine

    # Hit 1: higher retrieval score but off-taste; Hit 2: lower score, on-taste.
    high_score_off_taste = _make_hit(1, 0.9, vector=off_taste)
    low_score_on_taste = _make_hit(2, 0.6, vector=centroid[:])

    result = score_against_taste_with_vectors(
        [high_score_off_taste, low_score_on_taste], [off_taste, centroid[:]], profile=profile
    )

    # Taste must flip the order: the on-taste hit ranks first despite lower score.
    assert result[0].tmdb_id == 2
    assert result[0].taste_score > result[1].taste_score


def test_score_against_taste_with_vectors() -> None:
    profile = load_taste_profile(Path("data/taste_profile.json"))
    dim = len(profile.centroid)
    # Identical vector to centroid → taste_score ≈ 1.0
    hits = [_make_hit(1, 0.8)]
    vecs = [profile.centroid[:]]
    result = score_against_taste_with_vectors(hits, vecs, profile=profile)
    assert result[0].taste_score > 0.99


def test_score_against_taste_empty() -> None:
    profile = load_taste_profile(Path("data/taste_profile.json"))
    result = score_against_taste([], profile=profile)
    assert result == []
