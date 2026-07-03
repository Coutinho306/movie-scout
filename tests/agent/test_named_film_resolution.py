"""Unit tests for named-film resolution: recommend_similar, similar_movies_tool,
resolve_film wiring, and similar_movies in _build_rag_tools.

All Qdrant and TMDB HTTP calls are mocked — no live network required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from retrieval.config import RetrievalSettings
from retrieval.models import MovieFilters, MovieHit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_hit(tmdb_id: int, score: float = 0.9) -> MovieHit:
    return MovieHit(
        tmdb_id=tmdb_id,
        title=f"Film {tmdb_id}",
        year=2020,
        overview="Some overview",
        genres=["Drama"],
        vote_average=7.5,
        score=score,
    )


def _make_scored_point(tmdb_id: int, point_id: int, vector: list[float]) -> MagicMock:
    """Mimic a qdrant_client ScoredPoint with payload, id, and vector."""
    pt = MagicMock()
    pt.id = point_id
    pt.score = 0.85
    pt.payload = {
        "tmdb_id": tmdb_id,
        "title": f"Film {tmdb_id}",
        "year": 2020,
        "overview": "Test overview",
        "genres": ["Action"],
        "vote_average": 7.0,
    }
    pt.vector = vector
    return pt


def _make_record(point_id: int, vector: list[float]) -> MagicMock:
    """Mimic a qdrant_client Record returned by retrieve()."""
    rec = MagicMock()
    rec.id = point_id
    rec.vector = vector
    return rec


# ---------------------------------------------------------------------------
# recommend_similar tests
# ---------------------------------------------------------------------------


class TestRecommendSimilar:
    def test_returns_empty_when_seed_not_found(self) -> None:
        """When scroll returns no records, recommend_similar returns []."""
        from retrieval.movies import recommend_similar

        mock_client = MagicMock()
        mock_client.scroll.return_value = ([], None)

        settings = RetrievalSettings()
        with patch("retrieval.movies.get_qdrant_client", return_value=mock_client):
            result = recommend_similar(12345, settings=settings)

        assert result == []
        mock_client.retrieve.assert_not_called()

    def test_uses_fetched_seed_vector_for_query(self) -> None:
        """recommend_similar passes the seed's stored vector to query_points."""
        from retrieval.movies import recommend_similar

        seed_tmdb_id = 999
        seed_point_id = 42
        seed_vector = [0.1, 0.2, 0.3]

        scroll_record = MagicMock()
        scroll_record.id = seed_point_id

        seed_record = _make_record(seed_point_id, seed_vector)

        result_point = _make_scored_point(tmdb_id=111, point_id=100, vector=[0.4, 0.5, 0.6])

        mock_client = MagicMock()
        mock_client.scroll.return_value = ([scroll_record], None)
        mock_client.retrieve.return_value = [seed_record]
        mock_response = MagicMock()
        mock_response.points = [result_point]
        mock_client.query_points.return_value = mock_response

        settings = RetrievalSettings()
        with patch("retrieval.movies.get_qdrant_client", return_value=mock_client):
            hits = recommend_similar(seed_tmdb_id, settings=settings, k=5)

        # query_points was called
        mock_client.query_points.assert_called_once()
        call_kwargs = mock_client.query_points.call_args
        # The NearestQuery should wrap the seed's vector
        from qdrant_client.models import NearestQuery

        query_arg = call_kwargs.kwargs.get("query") or call_kwargs.args[1]
        assert isinstance(query_arg, NearestQuery)
        assert list(query_arg.nearest) == seed_vector

        assert len(hits) == 1
        assert hits[0].tmdb_id == 111

    def test_seed_always_excluded_from_results(self) -> None:
        """The seed's own tmdb_id must never appear in the returned hits."""
        from retrieval.movies import recommend_similar

        seed_tmdb_id = 500
        seed_point_id = 10

        scroll_record = MagicMock()
        scroll_record.id = seed_point_id
        seed_record = _make_record(seed_point_id, [0.1, 0.2])

        # Result points include the seed itself and another film
        point_seed = _make_scored_point(tmdb_id=seed_tmdb_id, point_id=10, vector=[0.1, 0.2])
        point_other = _make_scored_point(tmdb_id=200, point_id=20, vector=[0.3, 0.4])

        mock_client = MagicMock()
        mock_client.scroll.return_value = ([scroll_record], None)
        mock_client.retrieve.return_value = [seed_record]
        mock_response = MagicMock()
        mock_response.points = [point_seed, point_other]
        mock_client.query_points.return_value = mock_response

        settings = RetrievalSettings()
        with patch("retrieval.movies.get_qdrant_client", return_value=mock_client):
            hits = recommend_similar(seed_tmdb_id, settings=settings, k=5)

        ids = {h.tmdb_id for h in hits}
        assert seed_tmdb_id not in ids
        assert 200 in ids

    def test_exclude_tmdb_ids_from_filters_are_also_excluded(self) -> None:
        """Additional exclude_tmdb_ids in filters are merged with seed exclusion."""
        from retrieval.movies import recommend_similar

        seed_tmdb_id = 500
        extra_exclude = 300

        scroll_record = MagicMock()
        scroll_record.id = 10
        seed_record = _make_record(10, [0.1, 0.2])

        point_seed = _make_scored_point(tmdb_id=seed_tmdb_id, point_id=10, vector=[0.1, 0.2])
        point_excluded = _make_scored_point(tmdb_id=extra_exclude, point_id=30, vector=[0.5, 0.6])
        point_ok = _make_scored_point(tmdb_id=400, point_id=40, vector=[0.7, 0.8])

        mock_client = MagicMock()
        mock_client.scroll.return_value = ([scroll_record], None)
        mock_client.retrieve.return_value = [seed_record]
        mock_response = MagicMock()
        mock_response.points = [point_seed, point_excluded, point_ok]
        mock_client.query_points.return_value = mock_response

        filters = MovieFilters(exclude_tmdb_ids={extra_exclude})
        settings = RetrievalSettings()
        with patch("retrieval.movies.get_qdrant_client", return_value=mock_client):
            hits = recommend_similar(seed_tmdb_id, settings=settings, filters=filters, k=5)

        ids = {h.tmdb_id for h in hits}
        assert seed_tmdb_id not in ids
        assert extra_exclude not in ids
        assert 400 in ids


# ---------------------------------------------------------------------------
# similar_movies_tool tests
# ---------------------------------------------------------------------------


class TestSimilarMoviesTool:
    def test_seed_excluded_via_tool(self) -> None:
        """similar_movies_tool always adds seed to exclude_tmdb_ids before delegating."""
        from agent.tools.vector_search_movies import similar_movies_tool

        seed_id = 77
        returned_hit = _make_hit(tmdb_id=88)

        with patch("agent.tools.vector_search_movies.recommend_similar") as mock_rec:
            mock_rec.return_value = [returned_hit]
            hits = similar_movies_tool(seed_id, k=5)

        mock_rec.assert_called_once()
        _, call_kwargs = mock_rec.call_args[0], mock_rec.call_args[1]
        passed_filters: MovieFilters = call_kwargs["filters"]
        assert seed_id in passed_filters.exclude_tmdb_ids

        assert len(hits) == 1
        assert hits[0].tmdb_id == 88

    def test_seed_excluded_merges_caller_filters(self) -> None:
        """Caller-supplied exclude_tmdb_ids are merged, not replaced."""
        from agent.tools.vector_search_movies import similar_movies_tool

        seed_id = 77
        caller_exclude = {999}

        with patch("agent.tools.vector_search_movies.recommend_similar") as mock_rec:
            mock_rec.return_value = []
            similar_movies_tool(seed_id, filters=MovieFilters(exclude_tmdb_ids=caller_exclude), k=5)

        _, call_kwargs = mock_rec.call_args[0], mock_rec.call_args[1]
        passed_filters: MovieFilters = call_kwargs["filters"]
        assert seed_id in passed_filters.exclude_tmdb_ids
        assert 999 in passed_filters.exclude_tmdb_ids

    def test_returns_empty_when_seed_not_in_corpus(self) -> None:
        from agent.tools.vector_search_movies import similar_movies_tool

        with patch("agent.tools.vector_search_movies.recommend_similar", return_value=[]):
            result = similar_movies_tool(12345, k=5)

        assert result == []


# ---------------------------------------------------------------------------
# _build_rag_tools: search_movies is the only movie-search tool now (seed
# resolution collapsed into search_movies_tool itself, no separate
# resolve_film/similar_movies tools exposed to the LLM — see
# agent/tools/vector_search_movies.py and agent/tools/seed_film.py).
# ---------------------------------------------------------------------------


class TestBuildRagTools:
    def _get_tool_names(self, collected: list[dict] | None = None) -> list[str]:
        from agent.nodes.rag import _build_rag_tools

        tools = _build_rag_tools(collected or [], region="US", top_k=10)
        return [t.name for t in tools]

    def test_existing_tools_still_present(self) -> None:
        names = self._get_tool_names()
        for expected in ("search_movies", "search_reviews", "match_taste", "tmdb_lookup_providers"):
            assert expected in names

    def test_no_separate_resolve_or_similar_tools_exposed(self) -> None:
        """resolve_film/similar_movies are no longer separate agent-visible tools."""
        names = self._get_tool_names()
        assert "resolve_film" not in names
        assert "similar_movies" not in names


# ---------------------------------------------------------------------------
# seed_film extraction + search_movies_tool collapse
# ---------------------------------------------------------------------------


class TestSeedFilmExtraction:
    def test_extracts_seed_from_common_phrasings(self) -> None:
        from agent.tools.seed_film import extract_seed_title

        assert extract_seed_title("a film like Arrival") == "Arrival"
        assert (
            extract_seed_title("a film with the same theme as Glass Onion")
            == "Glass Onion"
        )
        assert extract_seed_title("movies like Glass Onion") == "Glass Onion"
        assert (
            extract_seed_title("a movie similar to The Prestige") == "The Prestige"
        )

    def test_returns_none_for_no_seed(self) -> None:
        from agent.tools.seed_film import extract_seed_title

        assert extract_seed_title("a dark psychological thriller about identity") is None

    def test_extracts_seed_from_rewritten_long_sentence(self) -> None:
        """retrieval/rewrite.py's query_rewrite expands short seed queries into
        long descriptive sentences — extraction must not swallow the trailing
        description as part of the title (regression: greedy end-of-string
        capture previously captured everything after 'Arrival')."""
        from agent.tools.seed_film import extract_seed_title

        rewritten = (
            "a thought-provoking science fiction film similar to Arrival, "
            "featuring complex narratives, themes of communication and time, "
            "and strong character development, preferably directed by Denis "
            "Villeneuve or with a similar artistic style."
        )
        assert extract_seed_title(rewritten) == "Arrival"


class TestSearchMoviesToolCollapse:
    def test_resolves_seed_and_delegates_to_similar_movies(self) -> None:
        """search_movies_tool resolves a named seed and uses similar_movies_tool."""
        from agent.tools.vector_search_movies import search_movies_tool

        hit = _make_hit(tmdb_id=546554)
        with patch(
            "agent.tools.vector_search_movies.search_tmdb", return_value=661374
        ) as mock_tmdb, patch(
            "agent.tools.vector_search_movies.similar_movies_tool",
            return_value=[hit],
        ) as mock_similar:
            hits = search_movies_tool("a film with the same theme as Glass Onion", k=5)

        mock_tmdb.assert_called_once_with("Glass Onion")
        mock_similar.assert_called_once()
        assert hits == [hit]

    def test_falls_back_to_text_search_when_no_seed_named(self) -> None:
        """No seed phrasing detected -> plain text search, no TMDB lookup."""
        from agent.tools.vector_search_movies import search_movies_tool

        hit = _make_hit(tmdb_id=1)
        with patch(
            "agent.tools.vector_search_movies.search_tmdb"
        ) as mock_tmdb, patch(
            "agent.tools.vector_search_movies.search_movies", return_value=[hit]
        ) as mock_search:
            hits = search_movies_tool("a dark psychological thriller about identity", k=5)

        mock_tmdb.assert_not_called()
        mock_search.assert_called_once()
        assert hits == [hit]

    def test_falls_back_to_text_search_when_seed_not_resolved(self) -> None:
        """Seed phrasing detected but TMDB lookup misses -> falls back to text search."""
        from agent.tools.vector_search_movies import search_movies_tool

        hit = _make_hit(tmdb_id=2)
        with patch(
            "agent.tools.vector_search_movies.search_tmdb", return_value=None
        ), patch(
            "agent.tools.vector_search_movies.search_movies", return_value=[hit]
        ) as mock_search:
            hits = search_movies_tool("a film like SomeUnknownFilm12345", k=5)

        mock_search.assert_called_once()
        assert hits == [hit]
