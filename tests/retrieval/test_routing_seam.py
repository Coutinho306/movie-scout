"""Unit tests for search_movies_tool routing seam and caller-precedence rule.

All tests mock out the downstream calls (search_movies, extract_seed_title,
extract_actor_name, search_tmdb) so no network or model calls are made.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from retrieval.config import RetrievalSettings
from retrieval.models import MovieHit


def _fake_hit(**kwargs: object) -> MovieHit:
    defaults: dict = dict(
        tmdb_id=1,
        title="Test Film",
        year=2020,
        overview="A test film.",
        genres=["Drama"],
        vote_average=7.0,
        score=0.9,
    )
    defaults.update(kwargs)
    return MovieHit(**defaults)


# Path-based patch targets
_SEED_PATH = "agent.tools.vector_search_movies.extract_seed_title"
_ACTOR_PATH = "agent.tools.vector_search_movies.extract_actor_name"
_SEARCH_PATH = "agent.tools.vector_search_movies.search_movies"
_CLASSIFY_PATH = "agent.tools.vector_search_movies.classify_query_mode"


def _no_seed_no_actor() -> tuple:
    """Return patch context managers for no-seed, no-actor branches."""
    return (
        patch(_SEED_PATH, return_value=None),
        patch(_ACTOR_PATH, return_value=None),
    )


class TestClassifierCalledOnFallthrough:
    """When no explicit settings are passed, the classifier must be called."""

    def test_classifier_called_for_dense_query(self) -> None:
        """Classifier called with the query text; its result sets hybrid."""
        with (
            patch(_SEED_PATH, return_value=None),
            patch(_ACTOR_PATH, return_value=None),
            patch(_CLASSIFY_PATH, return_value=False) as mock_classify,
            patch(_SEARCH_PATH, return_value=[_fake_hit()]) as mock_search,
        ):
            from agent.tools.vector_search_movies import search_movies_tool

            search_movies_tool("Gone Girl")

        mock_classify.assert_called_once_with("Gone Girl")
        # settings passed to search_movies must have hybrid=False
        _, call_kwargs = mock_search.call_args
        assert call_kwargs["settings"].hybrid is False

    def test_classifier_called_for_hybrid_query(self) -> None:
        """Classifier returning True → settings.hybrid=True passed to search_movies."""
        with (
            patch(_SEED_PATH, return_value=None),
            patch(_ACTOR_PATH, return_value=None),
            patch(_CLASSIFY_PATH, return_value=True) as mock_classify,
            patch(_SEARCH_PATH, return_value=[_fake_hit()]) as mock_search,
        ):
            from agent.tools.vector_search_movies import search_movies_tool

            search_movies_tool("a Drama, Thriller film — Edge of your seat.")

        mock_classify.assert_called_once()
        _, call_kwargs = mock_search.call_args
        assert call_kwargs["settings"].hybrid is True


class TestExplicitCallerPrecedence:
    """When explicit settings= is passed, the classifier must NOT be called."""

    def test_explicit_hybrid_true_respected(self) -> None:
        explicit = RetrievalSettings(hybrid=True)
        with (
            patch(_SEED_PATH, return_value=None),
            patch(_ACTOR_PATH, return_value=None),
            patch(_CLASSIFY_PATH) as mock_classify,
            patch(_SEARCH_PATH, return_value=[_fake_hit()]) as mock_search,
        ):
            from agent.tools.vector_search_movies import search_movies_tool

            search_movies_tool("abstract query", settings=explicit)

        mock_classify.assert_not_called()
        _, call_kwargs = mock_search.call_args
        assert call_kwargs["settings"].hybrid is True

    def test_explicit_hybrid_false_respected(self) -> None:
        explicit = RetrievalSettings(hybrid=False)
        with (
            patch(_SEED_PATH, return_value=None),
            patch(_ACTOR_PATH, return_value=None),
            patch(_CLASSIFY_PATH) as mock_classify,
            patch(_SEARCH_PATH, return_value=[_fake_hit()]) as mock_search,
        ):
            from agent.tools.vector_search_movies import search_movies_tool

            # Even a strongly hybrid-shaped query should be overridden
            search_movies_tool(
                "a Action, Thriller film — Buckle up.", settings=explicit
            )

        mock_classify.assert_not_called()
        _, call_kwargs = mock_search.call_args
        assert call_kwargs["settings"].hybrid is False

    def test_explicit_settings_other_fields_preserved(self) -> None:
        """Explicit settings preserve all fields (top_k, score_threshold, etc.)."""
        explicit = RetrievalSettings(hybrid=True, top_k=20, score_threshold=0.5)
        with (
            patch(_SEED_PATH, return_value=None),
            patch(_ACTOR_PATH, return_value=None),
            patch(_CLASSIFY_PATH) as mock_classify,
            patch(_SEARCH_PATH, return_value=[_fake_hit()]) as mock_search,
        ):
            from agent.tools.vector_search_movies import search_movies_tool

            search_movies_tool("some query", settings=explicit)

        mock_classify.assert_not_called()
        _, call_kwargs = mock_search.call_args
        s = call_kwargs["settings"]
        assert s.hybrid is True
        assert s.top_k == 20
        assert s.score_threshold == 0.5


class TestRouterNotInvokedForSeedOrActorPaths:
    """Classifier must not be called when seed-film or actor path fires."""

    def test_seed_path_bypasses_classifier(self) -> None:
        """Seed-film path fires → classifier never called."""
        with (
            patch(_SEED_PATH, return_value="Inception"),
            patch("agent.tools.vector_search_movies.search_tmdb", return_value=27205),
            patch(
                "agent.tools.vector_search_movies.similar_movies_tool",
                return_value=[_fake_hit()],
            ),
            patch(_CLASSIFY_PATH) as mock_classify,
        ):
            from agent.tools.vector_search_movies import search_movies_tool

            result = search_movies_tool("a film like Inception")

        mock_classify.assert_not_called()
        assert result  # got hits from seed path

    def test_actor_path_bypasses_classifier(self) -> None:
        """Actor path fires → classifier never called."""
        with (
            patch(_SEED_PATH, return_value=None),
            patch(_ACTOR_PATH, return_value="Tom Hanks"),
            patch(
                "agent.tools.vector_search_movies.list_movies_by_cast",
                return_value=[_fake_hit()],
            ),
            patch(_CLASSIFY_PATH) as mock_classify,
        ):
            from agent.tools.vector_search_movies import search_movies_tool

            result = search_movies_tool("films starring Tom Hanks")

        mock_classify.assert_not_called()
        assert result
