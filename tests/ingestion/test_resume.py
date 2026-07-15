"""AC1 + AC2 — Resume / skip-existing tests.

AC1: With --resume, the loader's per-film work (fetch_movie_metadata /
     embedder.embed_texts) is invoked only for the diffed missing ids and NOT
     for any already-present id.

AC2: Paginated scroll with a small page size correctly enumerates a seeded set
     spanning ≥2 pages (next_page_offset cursor followed to None).

All Qdrant / TMDB / OpenAI calls are mocked — no real network, no real collections.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest

from ingestion.resources.tmdb_movies import (
    _existing_tmdb_ids,
    load_tmdb_movies,
)
from ingestion.resources.tmdb_reviews import (
    _existing_review_tmdb_ids,
    load_tmdb_reviews,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_metadata(tmdb_id: int) -> MagicMock:
    """Return a minimal TmdbMovieMetadata-shaped mock."""
    meta = MagicMock()
    meta.tmdb_id = tmdb_id
    meta.title = f"Movie {tmdb_id}"
    meta.year = 2000
    meta.overview = "overview"
    meta.tagline = ""
    meta.genres = ["Drama"]
    meta.cast = ["Actor"]
    meta.director = "Director"
    meta.runtime = 90
    meta.vote_average = 7.0
    meta.popularity = 50.0
    meta.keywords = []
    meta.themes = ""
    meta.embed_text = f"embed text for {tmdb_id}"
    return meta


def _make_scroll_side_effect(
    all_ids: list[int], page_size: int
) -> list[tuple[list[MagicMock], Any]]:
    """Build the list of (records, next_offset) tuples that scroll() returns page-by-page."""
    pages = []
    for i in range(0, len(all_ids), page_size):
        chunk = all_ids[i : i + page_size]
        records = []
        for tid in chunk:
            rec = MagicMock()
            rec.payload = {"tmdb_id": tid}
            records.append(rec)
        is_last = (i + page_size) >= len(all_ids)
        next_offset = None if is_last else i + page_size
        pages.append((records, next_offset))
    # If all_ids is empty, scroll returns one empty page.
    if not pages:
        pages.append(([], None))
    return pages


# ---------------------------------------------------------------------------
# AC2 — _existing_tmdb_ids paginates correctly
# ---------------------------------------------------------------------------


def test_existing_tmdb_ids_single_page() -> None:
    """Single-page scroll: all ids collected, no offset follow-up."""
    seeded = [1, 2, 3]
    mock_client = MagicMock()
    mock_client.scroll.side_effect = _make_scroll_side_effect(seeded, page_size=10)

    result = _existing_tmdb_ids(mock_client, "test_collection")

    assert result == {1, 2, 3}
    assert mock_client.scroll.call_count == 1


def test_existing_tmdb_ids_multi_page() -> None:
    """Multi-page scroll: next_page_offset cursor is followed until None.

    Uses page_size=3 against a seeded set of 7 ids → 3 pages (3+3+1).
    Every seeded id must be collected.
    """
    seeded = list(range(10, 17))  # 7 ids: 10..16
    page_size = 3
    mock_client = MagicMock()
    mock_client.scroll.side_effect = _make_scroll_side_effect(seeded, page_size=page_size)

    # Patch _SCROLL_PAGE_SIZE so _existing_tmdb_ids uses our small page size.
    with patch("ingestion.resources.tmdb_movies._SCROLL_PAGE_SIZE", page_size):
        result = _existing_tmdb_ids(mock_client, "test_collection")

    assert result == set(seeded), f"Missing: {set(seeded) - result}"
    # 7 ids, page_size 3 → ceil(7/3) = 3 pages
    assert mock_client.scroll.call_count == 3


def test_existing_tmdb_ids_empty_collection() -> None:
    """Empty collection: returns empty set, one scroll call."""
    mock_client = MagicMock()
    mock_client.scroll.side_effect = _make_scroll_side_effect([], page_size=10)

    result = _existing_tmdb_ids(mock_client, "test_collection")

    assert result == set()
    assert mock_client.scroll.call_count == 1


# ---------------------------------------------------------------------------
# AC1 — load_tmdb_movies with resume=True skips present ids
# ---------------------------------------------------------------------------


def test_load_tmdb_movies_resume_skips_existing_ids() -> None:
    """With resume=True, fetch_movie_metadata is called ONLY for missing ids."""
    all_ids = [1, 2, 3, 4, 5]
    present_ids = [1, 3, 5]
    missing_ids = [2, 4]

    mock_qdrant = MagicMock()
    # scroll returns the present_ids on the first (and only) page
    present_records = []
    for tid in present_ids:
        rec = MagicMock()
        rec.payload = {"tmdb_id": tid}
        present_records.append(rec)
    mock_qdrant.scroll.return_value = (present_records, None)

    mock_embedder = MagicMock()
    mock_embedder.embed_texts.return_value = [[0.1, 0.2]]

    def _fake_fetch(tmdb_id: int, api_key: str, *, embed_text_recipe: str = "base"):
        return _make_metadata(tmdb_id)

    with (
        patch("ingestion.resources.tmdb_movies.QdrantClient", return_value=mock_qdrant),
        patch(
            "ingestion.resources.tmdb_movies.fetch_movie_metadata",
            side_effect=_fake_fetch,
        ) as mock_fetch,
    ):
        loaded = load_tmdb_movies(
            api_key="fake",
            qdrant_url="http://localhost",
            qdrant_api_key="fake",
            watched_tmdb_ids=set(),
            embedder=mock_embedder,
            collection_name="calib_test",
            explicit_tmdb_ids=all_ids,
            resume=True,
        )

    # Only missing ids were fetched.
    fetched_ids = {c.args[0] for c in mock_fetch.call_args_list}
    assert fetched_ids == set(missing_ids), (
        f"Expected fetch for {missing_ids}, got {sorted(fetched_ids)}"
    )
    assert loaded == len(missing_ids)


def test_load_tmdb_movies_no_resume_processes_all_ids() -> None:
    """With resume=False (default), all explicit ids are processed regardless of scroll."""
    all_ids = [10, 20, 30]

    mock_qdrant = MagicMock()
    # scroll should NOT be called when resume=False
    mock_embedder = MagicMock()
    mock_embedder.embed_texts.return_value = [[0.1, 0.2]]

    def _fake_fetch(tmdb_id: int, api_key: str, *, embed_text_recipe: str = "base"):
        return _make_metadata(tmdb_id)

    with (
        patch("ingestion.resources.tmdb_movies.QdrantClient", return_value=mock_qdrant),
        patch(
            "ingestion.resources.tmdb_movies.fetch_movie_metadata",
            side_effect=_fake_fetch,
        ) as mock_fetch,
    ):
        loaded = load_tmdb_movies(
            api_key="fake",
            qdrant_url="http://localhost",
            qdrant_api_key="fake",
            watched_tmdb_ids=set(),
            embedder=mock_embedder,
            collection_name="calib_test",
            explicit_tmdb_ids=all_ids,
            resume=False,  # default
        )

    assert mock_qdrant.scroll.call_count == 0, "scroll must not be called when resume=False"
    fetched_ids = {c.args[0] for c in mock_fetch.call_args_list}
    assert fetched_ids == set(all_ids)
    assert loaded == len(all_ids)


def test_load_tmdb_movies_resume_all_present_loads_nothing() -> None:
    """With resume=True and all ids present, no fetch and 0 loaded."""
    all_ids = [7, 8, 9]

    mock_qdrant = MagicMock()
    present_records = []
    for tid in all_ids:
        rec = MagicMock()
        rec.payload = {"tmdb_id": tid}
        present_records.append(rec)
    mock_qdrant.scroll.return_value = (present_records, None)

    mock_embedder = MagicMock()

    with (
        patch("ingestion.resources.tmdb_movies.QdrantClient", return_value=mock_qdrant),
        patch(
            "ingestion.resources.tmdb_movies.fetch_movie_metadata",
        ) as mock_fetch,
    ):
        loaded = load_tmdb_movies(
            api_key="fake",
            qdrant_url="http://localhost",
            qdrant_api_key="fake",
            watched_tmdb_ids=set(),
            embedder=mock_embedder,
            collection_name="calib_test",
            explicit_tmdb_ids=all_ids,
            resume=True,
        )

    mock_fetch.assert_not_called()
    assert loaded == 0


# ---------------------------------------------------------------------------
# AC2 (multi-page) — load_tmdb_movies with ≥2 scroll pages
# ---------------------------------------------------------------------------


def test_load_tmdb_movies_resume_multi_page_scroll() -> None:
    """Multi-page scroll correctly collects all present ids before diffing.

    20 present ids across 4 pages of 5, then a list of 25 candidates:
    the 5 absent ids should be the only ones fetched.
    """
    present_ids = list(range(1, 21))  # 20 present
    candidate_ids = list(range(1, 26))  # 25 candidates
    missing_ids = list(range(21, 26))   # 5 missing

    page_size = 5
    scroll_pages = _make_scroll_side_effect(present_ids, page_size=page_size)

    mock_qdrant = MagicMock()
    mock_qdrant.scroll.side_effect = scroll_pages

    mock_embedder = MagicMock()
    mock_embedder.embed_texts.return_value = [[0.1, 0.2]]

    def _fake_fetch(tmdb_id: int, api_key: str, *, embed_text_recipe: str = "base"):
        return _make_metadata(tmdb_id)

    with (
        patch("ingestion.resources.tmdb_movies.QdrantClient", return_value=mock_qdrant),
        patch("ingestion.resources.tmdb_movies._SCROLL_PAGE_SIZE", page_size),
        patch(
            "ingestion.resources.tmdb_movies.fetch_movie_metadata",
            side_effect=_fake_fetch,
        ) as mock_fetch,
    ):
        loaded = load_tmdb_movies(
            api_key="fake",
            qdrant_url="http://localhost",
            qdrant_api_key="fake",
            watched_tmdb_ids=set(),
            embedder=mock_embedder,
            collection_name="calib_test",
            explicit_tmdb_ids=candidate_ids,
            resume=True,
        )

    fetched_ids = {c.args[0] for c in mock_fetch.call_args_list}
    assert fetched_ids == set(missing_ids)
    assert loaded == len(missing_ids)
    # 20 ids at page_size=5 → 4 scroll calls
    assert mock_qdrant.scroll.call_count == 4


# ---------------------------------------------------------------------------
# Reviews — _existing_review_tmdb_ids and load_tmdb_reviews resume
# ---------------------------------------------------------------------------


def test_existing_review_tmdb_ids_multi_page() -> None:
    """Multi-page scroll for reviews collects all distinct tmdb_ids."""
    seeded = [100, 101, 102, 103, 104, 105, 106]  # 7 ids
    page_size = 3
    mock_client = MagicMock()
    mock_client.scroll.side_effect = _make_scroll_side_effect(seeded, page_size=page_size)

    with patch("ingestion.resources.tmdb_reviews._SCROLL_PAGE_SIZE", page_size):
        result = _existing_review_tmdb_ids(mock_client, "reviews_test")

    assert result == set(seeded)
    assert mock_client.scroll.call_count == 3  # ceil(7/3)


def test_load_tmdb_reviews_resume_skips_present_films() -> None:
    """With resume=True, films with any chunk present are skipped entirely."""
    candidate_ids = [200, 201, 202, 203]
    present_ids = [200, 202]
    missing_ids = [201, 203]

    mock_qdrant = MagicMock()
    present_records = []
    for tid in present_ids:
        rec = MagicMock()
        rec.payload = {"tmdb_id": tid}
        present_records.append(rec)
    mock_qdrant.scroll.return_value = (present_records, None)

    mock_embedder = MagicMock()

    with (
        patch("ingestion.resources.tmdb_reviews.QdrantClient", return_value=mock_qdrant),
        patch(
            "ingestion.resources.tmdb_reviews.fetch_reviews",
            return_value=[],
        ) as mock_fetch_reviews,
    ):
        loaded = load_tmdb_reviews(
            api_key="fake",
            qdrant_url="http://localhost",
            qdrant_api_key="fake",
            candidate_tmdb_ids=candidate_ids,
            embedder=mock_embedder,
            collection_name="calib_reviews",
            resume=True,
        )

    # fetch_reviews called only for missing ids
    fetched_ids = {c.args[0] for c in mock_fetch_reviews.call_args_list}
    assert fetched_ids == set(missing_ids)


def test_load_tmdb_reviews_no_resume_processes_all() -> None:
    """With resume=False (default), scroll is never called."""
    candidate_ids = [300, 301]
    mock_qdrant = MagicMock()
    mock_embedder = MagicMock()

    with (
        patch("ingestion.resources.tmdb_reviews.QdrantClient", return_value=mock_qdrant),
        patch(
            "ingestion.resources.tmdb_reviews.fetch_reviews",
            return_value=[],
        ) as mock_fetch_reviews,
    ):
        load_tmdb_reviews(
            api_key="fake",
            qdrant_url="http://localhost",
            qdrant_api_key="fake",
            candidate_tmdb_ids=candidate_ids,
            embedder=mock_embedder,
            collection_name="calib_reviews",
            resume=False,
        )

    assert mock_qdrant.scroll.call_count == 0
    fetched_ids = {c.args[0] for c in mock_fetch_reviews.call_args_list}
    assert fetched_ids == set(candidate_ids)
