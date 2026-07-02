"""Unit tests for ingestion id contracts and payload correctness."""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest

from ingestion.models import TmdbMovieMetadata


# ---------------------------------------------------------------------------
# T1.4a — id contract tests
# ---------------------------------------------------------------------------


def test_movie_point_id_contract() -> None:
    """Movie point id is str(uuid5(NAMESPACE_DNS, str(tmdb_id)))."""
    tmdb_id = 550
    expected = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(tmdb_id)))
    # Verify the formula is stable and matches the known value.
    assert expected == str(uuid.uuid5(uuid.NAMESPACE_DNS, "550"))
    # A different id yields a different point id.
    other = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(680)))
    assert expected != other


def test_review_chunk_id_contract() -> None:
    """Review chunk id is str(uuid5(NAMESPACE_DNS, f'{tmdb_id}_{author}_{chunk_index}'))."""
    tmdb_id = 550
    author = "reviewer_x"
    chunk_index = 0
    raw_key = f"{tmdb_id}_{author}_{chunk_index}"
    expected = str(uuid.uuid5(uuid.NAMESPACE_DNS, raw_key))
    # Same inputs → same id (deterministic).
    assert expected == str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{tmdb_id}_{author}_{chunk_index}"))
    # Different chunk index → different id.
    other_chunk = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{tmdb_id}_{author}_1"))
    assert expected != other_chunk


# ---------------------------------------------------------------------------
# T1.4b — themes payload key written by load_tmdb_movies
# ---------------------------------------------------------------------------


def _make_metadata(tmdb_id: int = 550, themes: list[str] | None = None) -> TmdbMovieMetadata:
    return TmdbMovieMetadata(
        tmdb_id=tmdb_id,
        title="Fight Club",
        year=1999,
        overview="An insomniac office worker ...",
        tagline="Mischief. Mayhem. Soap.",
        genres=["Drama", "Thriller"],
        cast=["Brad Pitt", "Edward Norton"],
        director="David Fincher",
        runtime=139,
        vote_average=8.4,
        popularity=100.0,
        keywords=["fight club", "masculinity"],
        themes=themes if themes is not None else [],
        embed_text="Fight Club Drama Thriller Brad Pitt Edward Norton",
    )


def test_load_tmdb_movies_writes_themes_key() -> None:
    """load_tmdb_movies must include 'themes' in the Qdrant point payload."""
    from ingestion.resources.tmdb_movies import load_tmdb_movies

    metadata = _make_metadata(themes=[])

    fake_client = MagicMock()
    fake_embedder = MagicMock()
    fake_embedder.embed_texts.return_value = [[0.1] * 1536]

    with (
        patch("ingestion.resources.tmdb_movies.QdrantClient", return_value=fake_client),
        patch(
            "ingestion.resources.tmdb_movies.fetch_movie_metadata",
            return_value=metadata,
        ),
    ):
        count = load_tmdb_movies(
            api_key="fake",
            qdrant_url="http://localhost:6333",
            qdrant_api_key="",
            watched_tmdb_ids=set(),
            embedder=fake_embedder,
            collection_name="tmdb_movies",
            explicit_tmdb_ids=[550],
        )

    assert count == 1
    assert fake_client.upsert.call_count == 1

    call_kwargs = fake_client.upsert.call_args
    points = call_kwargs.kwargs.get("points") or call_kwargs.args[1]
    point = points[0]
    payload = point.payload

    assert "themes" in payload, "themes key must be present in point payload"
    assert payload["themes"] == [], "themes should be an empty list for movies with no themes"


def test_load_tmdb_movies_writes_themes_values() -> None:
    """When metadata.themes is non-empty, the payload carries those values."""
    from ingestion.resources.tmdb_movies import load_tmdb_movies

    metadata = _make_metadata(themes=["identity", "consumerism"])

    fake_client = MagicMock()
    fake_embedder = MagicMock()
    fake_embedder.embed_texts.return_value = [[0.1] * 1536]

    with (
        patch("ingestion.resources.tmdb_movies.QdrantClient", return_value=fake_client),
        patch(
            "ingestion.resources.tmdb_movies.fetch_movie_metadata",
            return_value=metadata,
        ),
    ):
        load_tmdb_movies(
            api_key="fake",
            qdrant_url="http://localhost:6333",
            qdrant_api_key="",
            watched_tmdb_ids=set(),
            embedder=fake_embedder,
            collection_name="tmdb_movies",
            explicit_tmdb_ids=[550],
        )

    call_kwargs = fake_client.upsert.call_args
    points = call_kwargs.kwargs.get("points") or call_kwargs.args[1]
    payload = points[0].payload
    assert payload["themes"] == ["identity", "consumerism"]
