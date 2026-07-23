"""Tests for T1.1/T1.2/T1.3/T1.4 (Phase 1, 0022-keywords-payload-bm25-and-filter).

AC-1: freshly ingested point carries non-empty keywords payload; 'keywords'
      appears in the payload-index schema (create_payload_index called).
AC-3: build_sparse_text(..., keywords=["heist"]) contains the keyword token;
      the no-keywords call is unchanged in shape (no Keywords clause).
"""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

from ingestion.chunking import build_sparse_text
from ingestion.models import TmdbMovieMetadata


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_metadata(keywords: list[str] | None = None) -> TmdbMovieMetadata:
    return TmdbMovieMetadata(
        tmdb_id=680,
        title="Pulp Fiction",
        year=1994,
        overview="The lives of two mob hitmen, a boxer, a gangster and his wife intertwine.",
        tagline="You won't know the facts until you've seen the fiction.",
        genres=["Crime", "Drama"],
        cast=["John Travolta", "Uma Thurman", "Samuel L. Jackson"],
        director="Quentin Tarantino",
        runtime=154,
        vote_average=8.9,
        popularity=120.0,
        keywords=keywords if keywords is not None else [],
        embed_text="Pulp Fiction Crime Drama John Travolta Uma Thurman",
    )


# ---------------------------------------------------------------------------
# AC-1: keywords key in upserted payload
# ---------------------------------------------------------------------------


class TestKeywordsPayload:
    def test_upsert_payload_contains_keywords_key(self) -> None:
        """_process_movie must write 'keywords' into the Qdrant point payload."""
        from ingestion.resources.tmdb_movies import _process_movie

        meta = _make_metadata(keywords=["heist", "nonlinear narrative"])

        mock_client = MagicMock()
        mock_embedder = MagicMock()
        mock_embedder.embed_texts.return_value = [[0.1] * 1536]

        with patch("ingestion.resources.tmdb_movies.fetch_movie_metadata", return_value=meta):
            _process_movie(
                680,
                api_key="fake",
                embedder=mock_embedder,
                client=mock_client,
                collection_name="tmdb_movies",
                embed_text_recipe="base",
                sparse=False,
            )

        payload = mock_client.upsert.call_args.kwargs["points"][0].payload
        assert "keywords" in payload, "Payload must have a 'keywords' key"
        assert payload["keywords"] == ["heist", "nonlinear narrative"]

    def test_upsert_payload_keywords_non_empty_when_metadata_has_keywords(self) -> None:
        """A freshly ingested point with non-empty metadata.keywords yields non-empty payload."""
        from ingestion.resources.tmdb_movies import _process_movie

        meta = _make_metadata(keywords=["gangster", "cult film"])

        mock_client = MagicMock()
        mock_embedder = MagicMock()
        mock_embedder.embed_texts.return_value = [[0.1] * 1536]

        with patch("ingestion.resources.tmdb_movies.fetch_movie_metadata", return_value=meta):
            _process_movie(
                680,
                api_key="fake",
                embedder=mock_embedder,
                client=mock_client,
                collection_name="tmdb_movies",
                embed_text_recipe="base",
                sparse=False,
            )

        payload = mock_client.upsert.call_args.kwargs["points"][0].payload
        assert payload["keywords"], "keywords must be non-empty when metadata carries keywords"

    def test_upsert_payload_keywords_empty_list_when_none(self) -> None:
        """A freshly ingested point with no TMDB keywords writes an empty list (not None)."""
        from ingestion.resources.tmdb_movies import _process_movie

        meta = _make_metadata(keywords=[])

        mock_client = MagicMock()
        mock_embedder = MagicMock()
        mock_embedder.embed_texts.return_value = [[0.1] * 1536]

        with patch("ingestion.resources.tmdb_movies.fetch_movie_metadata", return_value=meta):
            _process_movie(
                680,
                api_key="fake",
                embedder=mock_embedder,
                client=mock_client,
                collection_name="tmdb_movies",
                embed_text_recipe="base",
                sparse=False,
            )

        payload = mock_client.upsert.call_args.kwargs["points"][0].payload
        assert "keywords" in payload
        assert payload["keywords"] == []


# ---------------------------------------------------------------------------
# AC-1: keywords index created in ensure_collections
# ---------------------------------------------------------------------------


class TestKeywordsPayloadIndex:
    def test_ensure_collections_creates_keywords_index(self) -> None:
        """ensure_collections must call create_payload_index for 'keywords' KEYWORD."""
        from ingestion.config import Settings
        from ingestion.pipeline import ensure_collections

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
        indexed_fields: set[str] = set()
        for c in calls:
            if len(c.args) >= 2:
                indexed_fields.add(c.args[1])
            else:
                indexed_fields.add(c.kwargs.get("field_name", ""))

        assert "keywords" in indexed_fields, (
            "ensure_collections must call create_payload_index for 'keywords'"
        )


# ---------------------------------------------------------------------------
# AC-3: build_sparse_text keywords clause
# ---------------------------------------------------------------------------


class TestBuildSparseTextKeywords:
    def test_keywords_clause_present_when_keywords_provided(self) -> None:
        """build_sparse_text must include 'Keywords: heist' when keywords=['heist']."""
        text = build_sparse_text(
            title="Heat",
            year=1995,
            genres=["Crime", "Drama", "Thriller"],
            director="Michael Mann",
            cast=["Al Pacino", "Robert De Niro", "Val Kilmer"],
            tagline="A Los Angeles crime saga.",
            overview="A group of professional bank robbers start to feel the heat.",
            keywords=["heist"],
        )
        assert "Keywords:" in text, "Sparse text must contain 'Keywords:' clause"
        assert "heist" in text, "Sparse text must contain the keyword token 'heist'"

    def test_keywords_clause_contains_all_tokens(self) -> None:
        """build_sparse_text includes every keyword token when multiple keywords provided."""
        text = build_sparse_text(
            title="Heat",
            year=1995,
            genres=["Crime"],
            director="Michael Mann",
            cast=["Al Pacino"],
            tagline="",
            overview="Bank robbers.",
            keywords=["heist", "cat and mouse", "los angeles"],
        )
        assert "heist" in text
        assert "cat and mouse" in text
        assert "los angeles" in text

    def test_no_keywords_clause_when_keywords_none(self) -> None:
        """build_sparse_text must not include a Keywords clause when keywords=None."""
        text = build_sparse_text(
            title="Heat",
            year=1995,
            genres=["Crime"],
            director="Michael Mann",
            cast=["Al Pacino"],
            tagline="",
            overview="Bank robbers.",
            keywords=None,
        )
        assert "Keywords:" not in text

    def test_no_keywords_clause_when_keywords_empty_list(self) -> None:
        """build_sparse_text must not include a Keywords clause when keywords=[]."""
        text = build_sparse_text(
            title="Heat",
            year=1995,
            genres=["Crime"],
            director="Michael Mann",
            cast=["Al Pacino"],
            tagline="",
            overview="Bank robbers.",
            keywords=[],
        )
        assert "Keywords:" not in text

    def test_shape_unchanged_when_no_keywords(self) -> None:
        """The no-keywords call produces the same output as before this change."""
        text_no_kw = build_sparse_text(
            title="Heat",
            year=1995,
            genres=["Crime"],
            director="Michael Mann",
            cast=["Al Pacino"],
            tagline="A Los Angeles crime saga.",
            overview="Bank robbers start to feel the heat.",
        )
        text_explicit_none = build_sparse_text(
            title="Heat",
            year=1995,
            genres=["Crime"],
            director="Michael Mann",
            cast=["Al Pacino"],
            tagline="A Los Angeles crime saga.",
            overview="Bank robbers start to feel the heat.",
            keywords=None,
        )
        assert text_no_kw == text_explicit_none, (
            "Omitting keywords= must produce the same output as keywords=None"
        )
        assert "Keywords:" not in text_no_kw

    def test_keywords_passed_through_sparse_ingest_call_site(self) -> None:
        """_process_movie (sparse=True) passes metadata.keywords into build_sparse_text."""
        from ingestion.resources.tmdb_movies import _process_movie

        meta = _make_metadata(keywords=["heist", "nonlinear narrative"])

        mock_client = MagicMock()
        mock_embedder = MagicMock()
        mock_embedder.embed_texts.return_value = [[0.1] * 1536]

        captured: dict[str, object] = {}

        original_build = build_sparse_text

        def capture_sparse_text(**kwargs: object) -> str:
            captured.update(kwargs)
            return original_build(**kwargs)  # type: ignore[arg-type]

        with (
            patch("ingestion.resources.tmdb_movies.fetch_movie_metadata", return_value=meta),
            patch("ingestion.resources.tmdb_movies.build_sparse_text", side_effect=capture_sparse_text),
        ):
            _process_movie(
                680,
                api_key="fake",
                embedder=mock_embedder,
                client=mock_client,
                collection_name="tmdb_movies",
                embed_text_recipe="base",
                sparse=True,
            )

        assert "keywords" in captured, "build_sparse_text must receive 'keywords' kwarg"
        assert captured["keywords"] == ["heist", "nonlinear narrative"]
