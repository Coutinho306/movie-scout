"""Tests for retrieval.taste.score_against_taste."""

from pathlib import Path

import pytest

from ingestion.scripts.compute_taste import load_taste_profile
from retrieval.models import MovieHit
from retrieval.taste import score_against_taste, score_against_taste_with_vectors


def _make_hit(tmdb_id: int, score: float) -> MovieHit:
    return MovieHit(
        tmdb_id=tmdb_id,
        title=f"Film {tmdb_id}",
        year=2000,
        overview="",
        genres=[],
        vote_average=7.0,
        score=score,
    )


def test_score_against_taste_sorts_descending() -> None:
    profile = load_taste_profile(Path("data/taste_profile.json"))
    hits = [_make_hit(1, 0.9), _make_hit(2, 0.5), _make_hit(3, 0.7)]
    result = score_against_taste(hits, profile=profile)
    assert len(result) == 3
    # blended_score = 0.5 * retrieval_score (taste_score=0 since no vector)
    scores = [h.blended_score for h in result]
    assert scores == sorted(scores, reverse=True)


def test_score_against_taste_with_vectors() -> None:
    profile = load_taste_profile(Path("data/taste_profile.json"))
    dim = len(profile.centroid)
    # Identical vector to centroid → taste_score ≈ 1.0
    hits = [_make_hit(1, 0.8)]
    vecs = [profile.centroid[:]]
    result = score_against_taste_with_vectors(hits, vecs, profile=profile)
    assert result[0].taste_score > 0.99


def test_score_against_taste_empty() -> None:
    result = score_against_taste([])
    assert result == []
