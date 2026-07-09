"""Unit tests for the shared sparse-text builder (AC-1, drift guard).

Assertions:
- build_sparse_text output equals build_movie_embed_text(metadata, recipe="base")
  for an identical set of fields (parity).
- Output contains no "Keywords:" clause (no keyword enrichment in sparse recipe).
- Cast is capped at top-5 even when more are supplied (matches dense base recipe).
"""

from __future__ import annotations

import pytest

from ingestion.chunking import build_movie_embed_text, build_sparse_text
from ingestion.models import TmdbMovieMetadata


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

def _make_metadata(*, extra_cast: bool = False, with_keywords: bool = False) -> TmdbMovieMetadata:
    cast = ["Cillian Murphy", "Emily Blunt", "Matt Damon", "Robert Downey Jr.", "Florence Pugh"]
    if extra_cast:
        cast = cast + ["Kenneth Branagh", "Rami Malek", "Josh Hartnett", "Alden Ehrenreich", "David Krumholtz"]
    return TmdbMovieMetadata(
        tmdb_id=872585,
        title="Oppenheimer",
        year=2023,
        overview="The story of J. Robert Oppenheimer and the Manhattan Project.",
        tagline="The world forever changes.",
        genres=["Drama", "History", "Thriller"],
        cast=cast,
        director="Christopher Nolan",
        runtime=180,
        vote_average=8.3,
        popularity=200.0,
        keywords=["nuclear", "manhattan project"] if with_keywords else [],
        embed_text="",
    )


# ---------------------------------------------------------------------------
# AC-1: parity — build_sparse_text == base dense recipe
# ---------------------------------------------------------------------------

def test_sparse_text_equals_base_dense_recipe() -> None:
    """build_sparse_text must produce the identical string as build_movie_embed_text(..., recipe='base')."""
    meta = _make_metadata()
    dense_base = build_movie_embed_text(meta, recipe="base")
    sparse = build_sparse_text(
        title=meta.title,
        year=meta.year,
        genres=meta.genres,
        director=meta.director,
        cast=meta.cast,
        tagline=meta.tagline,
        overview=meta.overview,
    )
    assert sparse == dense_base, (
        f"build_sparse_text diverged from base dense recipe.\n"
        f"dense_base: {dense_base!r}\n"
        f"sparse:     {sparse!r}"
    )


def test_sparse_text_no_keywords_clause() -> None:
    """build_sparse_text must never contain a 'Keywords:' clause."""
    meta = _make_metadata(with_keywords=True)
    sparse = build_sparse_text(
        title=meta.title,
        year=meta.year,
        genres=meta.genres,
        director=meta.director,
        cast=meta.cast,
        tagline=meta.tagline,
        overview=meta.overview,
    )
    assert "Keywords:" not in sparse, (
        f"Sparse text must not contain a Keywords clause: {sparse!r}"
    )


def test_sparse_text_cast_capped_at_five() -> None:
    """build_sparse_text uses only top-5 cast members, matching the dense base recipe."""
    meta = _make_metadata(extra_cast=True)
    assert len(meta.cast) == 10, "fixture should have 10 cast members"

    sparse = build_sparse_text(
        title=meta.title,
        year=meta.year,
        genres=meta.genres,
        director=meta.director,
        cast=meta.cast,
        tagline=meta.tagline,
        overview=meta.overview,
    )
    dense_base = build_movie_embed_text(meta, recipe="base")

    # Both should cap at the same 5 actors.
    assert sparse == dense_base
    # The 6th actor should not appear in either.
    sixth_actor = meta.cast[5]
    assert sixth_actor not in sparse, f"6th cast member leaked into sparse text: {sixth_actor!r}"


def test_sparse_text_contains_genre_and_cast_tokens() -> None:
    """build_sparse_text must index genre and cast tokens for lexical BM25 matching."""
    meta = _make_metadata()
    sparse = build_sparse_text(
        title=meta.title,
        year=meta.year,
        genres=meta.genres,
        director=meta.director,
        cast=meta.cast,
        tagline=meta.tagline,
        overview=meta.overview,
    )
    for genre in meta.genres:
        assert genre in sparse, f"Genre {genre!r} absent from sparse text"
    for actor in meta.cast[:5]:
        assert actor in sparse, f"Actor {actor!r} absent from sparse text"
    assert meta.director in sparse, f"Director {meta.director!r} absent from sparse text"
    assert meta.title in sparse, f"Title {meta.title!r} absent from sparse text"
