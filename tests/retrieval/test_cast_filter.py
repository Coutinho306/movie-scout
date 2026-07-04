"""Unit tests for cast-filter and exact-title lookup functionality.

Covers:
- _build_filter: cast condition construction
- list_movies_by_cast: uses cast filter via scroll, NOT dense embedding (mocked Qdrant)
- find_by_exact_title: returns the full match set for a mocked multi-result scroll
- ensure_collections: issues the new cast and title index creation calls
"""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from retrieval.config import RetrievalSettings
from retrieval.models import MovieFilters, MovieHit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_record(
    point_id: int,
    tmdb_id: int,
    title: str = "Test Film",
    year: int = 2020,
    cast: list[str] | None = None,
) -> MagicMock:
    """Mimic a qdrant_client Record returned by scroll()."""
    rec = MagicMock()
    rec.id = point_id
    rec.payload = {
        "tmdb_id": tmdb_id,
        "title": title,
        "year": year,
        "overview": "A test film.",
        "genres": ["Drama"],
        "vote_average": 7.0,
        "cast": cast or [],
    }
    return rec


# ---------------------------------------------------------------------------
# _build_filter cast condition
# ---------------------------------------------------------------------------


class TestBuildFilterCast:
    def test_cast_filter_added_when_cast_set(self) -> None:
        """_build_filter includes a MatchAny cast condition when filters.cast is set."""
        from qdrant_client.models import FieldCondition, MatchAny

        from retrieval.movies import _build_filter

        f = _build_filter(MovieFilters(cast=["Ryan Gosling"]))
        assert f is not None
        assert f.must is not None

        cast_conditions = [
            c for c in f.must
            if isinstance(c, FieldCondition) and c.key == "cast"
        ]
        assert len(cast_conditions) == 1
        cond = cast_conditions[0]
        assert isinstance(cond.match, MatchAny)
        assert "Ryan Gosling" in cond.match.any

    def test_cast_filter_multi_actor(self) -> None:
        """_build_filter supports multiple actors via MatchAny."""
        from qdrant_client.models import FieldCondition, MatchAny

        from retrieval.movies import _build_filter

        f = _build_filter(MovieFilters(cast=["Ryan Gosling", "Emma Stone"]))
        assert f is not None
        cast_conditions = [
            c for c in f.must
            if isinstance(c, FieldCondition) and c.key == "cast"
        ]
        assert len(cast_conditions) == 1
        assert set(cast_conditions[0].match.any) == {"Ryan Gosling", "Emma Stone"}

    def test_no_cast_filter_when_cast_none(self) -> None:
        """_build_filter does not add a cast condition when filters.cast is None."""
        from qdrant_client.models import FieldCondition

        from retrieval.movies import _build_filter

        f = _build_filter(MovieFilters(year_min=2000))
        if f is None:
            return  # no conditions at all — also fine
        cast_conditions = [
            c for c in (f.must or [])
            if isinstance(c, FieldCondition) and c.key == "cast"
        ]
        assert len(cast_conditions) == 0

    def test_cast_filter_combined_with_genre(self) -> None:
        """cast and genres conditions both appear when both fields are set."""
        from qdrant_client.models import FieldCondition

        from retrieval.movies import _build_filter

        f = _build_filter(MovieFilters(cast=["Tom Hanks"], genres=["Drama"]))
        assert f is not None
        keys = [c.key for c in f.must if isinstance(c, FieldCondition)]
        assert "cast" in keys
        assert "genres" in keys


# ---------------------------------------------------------------------------
# list_movies_by_cast — uses scroll, not dense embedding
# ---------------------------------------------------------------------------


class TestListMoviesByCast:
    def test_uses_scroll_not_query_points(self) -> None:
        """list_movies_by_cast must call scroll, never query_points (no dense vector)."""
        from retrieval.movies import list_movies_by_cast

        record = _make_record(1, tmdb_id=550, title="Fight Club")
        mock_client = MagicMock()
        # First call returns one record and no next_offset.
        mock_client.scroll.return_value = ([record], None)

        settings = RetrievalSettings()
        with patch("retrieval.movies.get_qdrant_client", return_value=mock_client):
            hits = list_movies_by_cast("Brad Pitt", settings=settings, k=10)

        mock_client.scroll.assert_called_once()
        mock_client.query_points.assert_not_called()
        assert len(hits) == 1
        assert hits[0].tmdb_id == 550

    def test_passes_cast_filter_to_scroll(self) -> None:
        """list_movies_by_cast passes a MatchAny cast filter to scroll."""
        from qdrant_client.models import FieldCondition, MatchAny

        from retrieval.movies import list_movies_by_cast

        mock_client = MagicMock()
        mock_client.scroll.return_value = ([], None)

        settings = RetrievalSettings()
        with patch("retrieval.movies.get_qdrant_client", return_value=mock_client):
            list_movies_by_cast("Ryan Gosling", settings=settings, k=5)

        scroll_kwargs = mock_client.scroll.call_args.kwargs
        scroll_filter = scroll_kwargs.get("scroll_filter")
        assert scroll_filter is not None

        cast_conditions = [
            c for c in scroll_filter.must
            if isinstance(c, FieldCondition) and c.key == "cast"
        ]
        assert len(cast_conditions) == 1
        assert isinstance(cast_conditions[0].match, MatchAny)
        assert "Ryan Gosling" in cast_conditions[0].match.any

    def test_results_have_score_zero(self) -> None:
        """Scroll records lack a score; list_movies_by_cast sets score=0.0."""
        from retrieval.movies import list_movies_by_cast

        record = _make_record(1, tmdb_id=123, title="Some Film")
        mock_client = MagicMock()
        mock_client.scroll.return_value = ([record], None)

        settings = RetrievalSettings()
        with patch("retrieval.movies.get_qdrant_client", return_value=mock_client):
            hits = list_movies_by_cast("Some Actor", settings=settings, k=10)

        assert hits[0].score == 0.0

    def test_respects_k_cap(self) -> None:
        """list_movies_by_cast returns at most k results."""
        from retrieval.movies import list_movies_by_cast

        records = [_make_record(i, tmdb_id=i + 1) for i in range(20)]
        mock_client = MagicMock()
        # Return all 20 in one page; scroll reports no next page.
        mock_client.scroll.return_value = (records[:5], None)

        settings = RetrievalSettings()
        with patch("retrieval.movies.get_qdrant_client", return_value=mock_client):
            hits = list_movies_by_cast("Actor Name", settings=settings, k=5)

        assert len(hits) <= 5

    def test_paginates_across_multiple_scroll_pages(self) -> None:
        """list_movies_by_cast follows scroll pagination until next_offset is None."""
        from retrieval.movies import list_movies_by_cast

        page1 = [_make_record(1, tmdb_id=1)]
        page2 = [_make_record(2, tmdb_id=2)]

        mock_client = MagicMock()
        # First call returns page1 with a next_offset token; second call returns page2 with None.
        mock_client.scroll.side_effect = [
            (page1, "some_offset_token"),
            (page2, None),
        ]

        settings = RetrievalSettings()
        with patch("retrieval.movies.get_qdrant_client", return_value=mock_client):
            hits = list_movies_by_cast("Actor Name", settings=settings, k=10)

        assert mock_client.scroll.call_count == 2
        assert {h.tmdb_id for h in hits} == {1, 2}


# ---------------------------------------------------------------------------
# find_by_exact_title — returns the full match set
# ---------------------------------------------------------------------------


class TestFindByExactTitle:
    def test_returns_all_matching_records(self) -> None:
        """find_by_exact_title returns every record the scroll yields."""
        from retrieval.movies import find_by_exact_title

        records = [
            _make_record(1, tmdb_id=5155, title="Obsession", year=1943),
            _make_record(2, tmdb_id=4780, title="Obsession", year=1976),
            _make_record(3, tmdb_id=332672, title="Obsession", year=2015),
            _make_record(4, tmdb_id=1339713, title="Obsession", year=2026),
        ]
        mock_client = MagicMock()
        mock_client.scroll.return_value = (records, None)

        settings = RetrievalSettings()
        with patch("retrieval.movies.get_qdrant_client", return_value=mock_client):
            hits = find_by_exact_title("Obsession", settings=settings)

        assert len(hits) == 4
        tmdb_ids = {h.tmdb_id for h in hits}
        assert tmdb_ids == {5155, 4780, 332672, 1339713}

    def test_uses_match_value_not_match_any(self) -> None:
        """find_by_exact_title uses MatchValue (exact match) not MatchAny."""
        from qdrant_client.models import FieldCondition, MatchValue

        from retrieval.movies import find_by_exact_title

        mock_client = MagicMock()
        mock_client.scroll.return_value = ([], None)

        settings = RetrievalSettings()
        with patch("retrieval.movies.get_qdrant_client", return_value=mock_client):
            find_by_exact_title("Obsession", settings=settings)

        scroll_kwargs = mock_client.scroll.call_args.kwargs
        title_filter = scroll_kwargs.get("scroll_filter")
        assert title_filter is not None
        title_conditions = [
            c for c in title_filter.must
            if isinstance(c, FieldCondition) and c.key == "title"
        ]
        assert len(title_conditions) == 1
        assert isinstance(title_conditions[0].match, MatchValue)
        assert title_conditions[0].match.value == "Obsession"

    def test_does_not_call_query_points(self) -> None:
        """find_by_exact_title must use scroll only, never vector search."""
        from retrieval.movies import find_by_exact_title

        mock_client = MagicMock()
        mock_client.scroll.return_value = ([], None)

        settings = RetrievalSettings()
        with patch("retrieval.movies.get_qdrant_client", return_value=mock_client):
            find_by_exact_title("Obsession", settings=settings)

        mock_client.query_points.assert_not_called()

    def test_returns_empty_for_no_match(self) -> None:
        from retrieval.movies import find_by_exact_title

        mock_client = MagicMock()
        mock_client.scroll.return_value = ([], None)

        settings = RetrievalSettings()
        with patch("retrieval.movies.get_qdrant_client", return_value=mock_client):
            hits = find_by_exact_title("NoSuchFilm12345", settings=settings)

        assert hits == []

    def test_scores_are_zero(self) -> None:
        from retrieval.movies import find_by_exact_title

        records = [_make_record(1, tmdb_id=5155, title="Obsession", year=1943)]
        mock_client = MagicMock()
        mock_client.scroll.return_value = (records, None)

        settings = RetrievalSettings()
        with patch("retrieval.movies.get_qdrant_client", return_value=mock_client):
            hits = find_by_exact_title("Obsession", settings=settings)

        assert all(h.score == 0.0 for h in hits)

    def test_paginates_until_no_next_offset(self) -> None:
        """find_by_exact_title follows pagination to collect all matches."""
        from retrieval.movies import find_by_exact_title

        page1 = [_make_record(1, tmdb_id=1, title="X")]
        page2 = [_make_record(2, tmdb_id=2, title="X")]

        mock_client = MagicMock()
        mock_client.scroll.side_effect = [
            (page1, "offset_1"),
            (page2, None),
        ]

        settings = RetrievalSettings()
        with patch("retrieval.movies.get_qdrant_client", return_value=mock_client):
            hits = find_by_exact_title("X", settings=settings)

        assert mock_client.scroll.call_count == 2
        assert len(hits) == 2


# ---------------------------------------------------------------------------
# ensure_collections: cast and title index creation
# ---------------------------------------------------------------------------


class TestEnsureCollectionsCastIndex:
    def test_creates_cast_payload_index(self) -> None:
        """ensure_collections must call create_payload_index for 'cast' KEYWORD."""
        from qdrant_client.models import PayloadSchemaType

        from ingestion.pipeline import ensure_collections
        from ingestion.config import Settings

        mock_client = MagicMock()
        # Pretend collections already exist so no create_collection call needed.
        mock_collection = MagicMock()
        mock_collection.name = "tmdb_movies"
        mock_reviews = MagicMock()
        mock_reviews.name = "tmdb_reviews"
        mock_client.get_collections.return_value = MagicMock(
            collections=[mock_collection, mock_reviews]
        )

        settings = Settings()
        ensure_collections(mock_client, settings)

        # Collect all create_payload_index calls.
        calls = mock_client.create_payload_index.call_args_list
        indexed_fields = {c.args[1] if c.args else c.kwargs.get("field_name") for c in calls}
        # Also handle positional + keyword mixed calls.
        indexed_fields_all = set()
        for c in calls:
            if len(c.args) >= 2:
                indexed_fields_all.add(c.args[1])
            else:
                indexed_fields_all.add(c.kwargs.get("field_name", ""))

        assert "cast" in indexed_fields_all, "cast payload index must be created"

    def test_creates_title_payload_index(self) -> None:
        """ensure_collections must call create_payload_index for 'title' KEYWORD."""
        from ingestion.pipeline import ensure_collections
        from ingestion.config import Settings

        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.name = "tmdb_movies"
        mock_reviews = MagicMock()
        mock_reviews.name = "tmdb_reviews"
        mock_client.get_collections.return_value = MagicMock(
            collections=[mock_collection, mock_reviews]
        )

        settings = Settings()
        ensure_collections(mock_client, settings)

        calls = mock_client.create_payload_index.call_args_list
        indexed_fields_all = set()
        for c in calls:
            if len(c.args) >= 2:
                indexed_fields_all.add(c.args[1])
            else:
                indexed_fields_all.add(c.kwargs.get("field_name", ""))

        assert "title" in indexed_fields_all, "title payload index must be created"
