"""Unit tests for cast-filter routing, actor extraction, backfill script,
and same-title collision regression.

Covers:
- extract_actor_name precision/fallback
- extract_actor_name does NOT collide with extract_seed_title trigger phrasings
- search_movies_tool: actor phrasing dispatches to list_movies_by_cast, not dense search
- backfill: skip predicate (≥15 cast entries → skip, no TMDB/Qdrant write)
- backfill: calls set_payload, not upsert or overwrite_payload
- Regression (0013 AC-9): unique-title inform queries still answer normally after
  _supplement_collision_hits removal; collision disambiguation is pre-graph only.
"""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from retrieval.models import MovieHit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_hit(tmdb_id: int, title: str = "Test Film", year: int = 2020) -> MovieHit:
    return MovieHit(
        tmdb_id=tmdb_id,
        title=title,
        year=year,
        overview="Some overview",
        genres=["Drama"],
        vote_average=7.5,
        score=0.0,
    )


# ---------------------------------------------------------------------------
# extract_actor_name — precision-biased matching
# ---------------------------------------------------------------------------


class TestExtractActorName:
    def test_films_with_actor(self) -> None:
        from agent.tools.actor_film import extract_actor_name

        assert extract_actor_name("films with Ryan Gosling") == "Ryan Gosling"
        assert extract_actor_name("movies with Ryan Gosling") == "Ryan Gosling"

    def test_films_with_the_actor(self) -> None:
        from agent.tools.actor_film import extract_actor_name

        assert extract_actor_name("films with the actor Keanu Reeves") == "Keanu Reeves"
        assert extract_actor_name("movies with the actor Brad Pitt") == "Brad Pitt"

    def test_starring_phrasing(self) -> None:
        from agent.tools.actor_film import extract_actor_name

        assert extract_actor_name("starring Tom Hanks") == "Tom Hanks"
        assert extract_actor_name("movies starring Meryl Streep") == "Meryl Streep"

    def test_with_the_actor_standalone(self) -> None:
        from agent.tools.actor_film import extract_actor_name

        assert extract_actor_name("with the actor Cate Blanchett") == "Cate Blanchett"
        assert extract_actor_name("the actor Denzel Washington") == "Denzel Washington"

    def test_returns_none_for_no_actor_phrasing(self) -> None:
        from agent.tools.actor_film import extract_actor_name

        assert extract_actor_name("a dark psychological thriller about identity") is None
        assert extract_actor_name("science fiction movies") is None
        assert extract_actor_name("") is None

    def test_leading_name_returns_none(self) -> None:
        """Leading-name forms are NOT covered — must degrade to dense search."""
        from agent.tools.actor_film import extract_actor_name

        # "Keanu Reeves movies" has no supported trigger phrase — returns None.
        assert extract_actor_name("Keanu Reeves movies") is None
        assert extract_actor_name("Ryan Gosling films") is None

    def test_stops_at_clause_boundary(self) -> None:
        """Actor name capture must not swallow trailing clause content."""
        from agent.tools.actor_film import extract_actor_name

        name = extract_actor_name("films with Leonardo DiCaprio, preferably from the 1990s")
        assert name == "Leonardo DiCaprio"

    def test_multi_word_names(self) -> None:
        from agent.tools.actor_film import extract_actor_name

        assert extract_actor_name("films with Jean-Claude Van Damme") == "Jean-Claude Van Damme"
        assert extract_actor_name("starring Robert De Niro") == "Robert De Niro"


# ---------------------------------------------------------------------------
# Cross-extractor collision check
# extract_actor_name must NOT fire on extract_seed_title trigger phrasings, and vice versa.
# ---------------------------------------------------------------------------


SEED_PHRASINGS = [
    "a film like Arrival",
    "movies like Glass Onion",
    "a film with the same theme as The Prestige",
    "similar to Inception",
    "in the style of Kubrick",
    "a movie similar to Fight Club",
    "same vibe as Drive",
]

ACTOR_PHRASINGS = [
    "films with Ryan Gosling",
    "movies with the actor Keanu Reeves",
    "starring Brad Pitt",
    "with the actor Cate Blanchett",
    "films with the actor Tom Hanks",
]


class TestExtractorCrossCollision:
    @pytest.mark.parametrize("phrasing", SEED_PHRASINGS)
    def test_actor_extractor_does_not_fire_on_seed_phrasings(self, phrasing: str) -> None:
        """extract_actor_name must return None for all seed-film phrasings."""
        from agent.tools.actor_film import extract_actor_name

        result = extract_actor_name(phrasing)
        assert result is None, (
            f"extract_actor_name fired on seed phrasing '{phrasing}', returned '{result}'"
        )

    @pytest.mark.parametrize("phrasing", ACTOR_PHRASINGS)
    def test_seed_extractor_does_not_fire_on_actor_phrasings(self, phrasing: str) -> None:
        """extract_seed_title must return None for all actor phrasings."""
        from agent.tools.seed_film import extract_seed_title

        result = extract_seed_title(phrasing)
        assert result is None, (
            f"extract_seed_title fired on actor phrasing '{phrasing}', returned '{result}'"
        )


# ---------------------------------------------------------------------------
# search_movies_tool routing for actor phrasings
# ---------------------------------------------------------------------------


class TestSearchMoviesToolActorRouting:
    def test_actor_phrasing_dispatches_to_list_movies_by_cast(self) -> None:
        """Actor phrasing bypasses dense search and calls list_movies_by_cast."""
        from agent.tools.vector_search_movies import search_movies_tool

        hits = [_make_hit(100), _make_hit(101)]
        with (
            patch("agent.tools.vector_search_movies.list_movies_by_cast", return_value=hits) as mock_cast,
            patch("agent.tools.vector_search_movies.search_movies") as mock_dense,
            patch("agent.tools.vector_search_movies.search_tmdb") as mock_tmdb,
        ):
            result = search_movies_tool("films with Ryan Gosling", k=10)

        mock_cast.assert_called_once()
        call_kwargs = mock_cast.call_args
        # First positional arg is the actor name.
        actor_arg = call_kwargs.args[0] if call_kwargs.args else call_kwargs.kwargs.get("actor", "")
        assert actor_arg == "Ryan Gosling"
        mock_dense.assert_not_called()
        assert result == hits

    def test_actor_phrasing_does_not_call_dense_search(self) -> None:
        """Dense search must not be called when an actor phrasing is detected."""
        from agent.tools.vector_search_movies import search_movies_tool

        with (
            patch("agent.tools.vector_search_movies.list_movies_by_cast", return_value=[]),
            patch("agent.tools.vector_search_movies.search_movies") as mock_dense,
            patch("agent.tools.vector_search_movies.search_tmdb"),
        ):
            search_movies_tool("starring Keanu Reeves", k=10)

        mock_dense.assert_not_called()

    def test_non_actor_phrasing_falls_through_to_dense_search(self) -> None:
        """Queries without an actor trigger must still fall through to dense search."""
        from agent.tools.vector_search_movies import search_movies_tool

        hit = _make_hit(1)
        with (
            patch("agent.tools.vector_search_movies.list_movies_by_cast") as mock_cast,
            patch("agent.tools.vector_search_movies.search_movies", return_value=[hit]) as mock_dense,
            patch("agent.tools.vector_search_movies.search_tmdb", return_value=None),
        ):
            result = search_movies_tool("a dark thriller about revenge", k=10)

        mock_cast.assert_not_called()
        mock_dense.assert_called_once()
        assert result == [hit]

    def test_seed_phrasing_is_tried_before_actor_extraction(self) -> None:
        """Seed-film extraction (step 1) runs before actor extraction (step 2)."""
        from agent.tools.vector_search_movies import search_movies_tool

        # "a film like The Actor" would match both extractors in theory, but
        # seed_film extraction is tried first — confirm TMDB lookup is attempted.
        with (
            patch("agent.tools.vector_search_movies.search_tmdb", return_value=None) as mock_tmdb,
            patch("agent.tools.vector_search_movies.list_movies_by_cast", return_value=[]) as mock_cast,
            patch("agent.tools.vector_search_movies.search_movies", return_value=[]),
        ):
            search_movies_tool("a film like Arrival", k=10)

        # seed_film extraction fires first → TMDB lookup attempted.
        mock_tmdb.assert_called_once_with("Arrival")


# ---------------------------------------------------------------------------
# Backfill script: skip predicate and set_payload (not upsert/overwrite)
# ---------------------------------------------------------------------------


class TestBackfillCastPayload:
    def _make_scroll_record(
        self,
        point_id: int,
        tmdb_id: int,
        cast: list[str],
    ) -> MagicMock:
        rec = MagicMock()
        rec.id = point_id
        rec.payload = {"tmdb_id": tmdb_id, "cast": cast}
        return rec

    def test_skip_predicate_fires_for_15_or_more_cast(self) -> None:
        """Records with ≥15 stored cast entries must be skipped without TMDB call."""
        from scripts.backfill_cast_payload import backfill

        large_cast = [f"Actor {i}" for i in range(15)]
        record = self._make_scroll_record(1, tmdb_id=550, cast=large_cast)

        mock_client = MagicMock()
        mock_client.scroll.return_value = ([record], None)

        with (
            patch("scripts.backfill_cast_payload.QdrantClient", return_value=mock_client),
            patch("scripts.backfill_cast_payload._fetch_cast_top15") as mock_fetch,
        ):
            backfill(
                qdrant_url="http://localhost:6333",
                qdrant_api_key="",
                tmdb_api_key="fake_key",
                collection_name="tmdb_movies",
            )

        # TMDB should NOT be called for the skipped record.
        mock_fetch.assert_not_called()
        mock_client.set_payload.assert_not_called()

    def test_records_with_fewer_than_15_cast_are_updated(self) -> None:
        """Records with <15 cast entries should trigger a TMDB fetch + set_payload."""
        from scripts.backfill_cast_payload import backfill

        small_cast = ["Actor 1", "Actor 2"]
        record = self._make_scroll_record(1, tmdb_id=550, cast=small_cast)
        new_cast = [f"Actor {i}" for i in range(15)]

        mock_client = MagicMock()
        mock_client.scroll.return_value = ([record], None)

        with (
            patch("scripts.backfill_cast_payload.QdrantClient", return_value=mock_client),
            patch("scripts.backfill_cast_payload._fetch_cast_top15", return_value=new_cast),
        ):
            backfill(
                qdrant_url="http://localhost:6333",
                qdrant_api_key="",
                tmdb_api_key="fake_key",
                collection_name="tmdb_movies",
            )

        mock_client.set_payload.assert_called_once()

    def test_uses_set_payload_not_upsert(self) -> None:
        """Backfill must call set_payload (payload-only merge), never upsert."""
        from scripts.backfill_cast_payload import backfill

        record = self._make_scroll_record(1, tmdb_id=550, cast=["One Actor"])
        new_cast = [f"Actor {i}" for i in range(15)]

        mock_client = MagicMock()
        mock_client.scroll.return_value = ([record], None)

        with (
            patch("scripts.backfill_cast_payload.QdrantClient", return_value=mock_client),
            patch("scripts.backfill_cast_payload._fetch_cast_top15", return_value=new_cast),
        ):
            backfill(
                qdrant_url="http://localhost:6333",
                qdrant_api_key="",
                tmdb_api_key="fake_key",
                collection_name="tmdb_movies",
            )

        mock_client.upsert.assert_not_called()
        mock_client.overwrite_payload.assert_not_called()
        mock_client.set_payload.assert_called_once()
        set_call_kwargs = mock_client.set_payload.call_args.kwargs
        assert "cast" in set_call_kwargs.get("payload", {})

    def test_set_payload_only_updates_cast_key(self) -> None:
        """The payload dict passed to set_payload must only contain 'cast'."""
        from scripts.backfill_cast_payload import backfill

        record = self._make_scroll_record(1, tmdb_id=550, cast=[])
        new_cast = [f"Actor {i}" for i in range(10)]

        mock_client = MagicMock()
        mock_client.scroll.return_value = ([record], None)

        with (
            patch("scripts.backfill_cast_payload.QdrantClient", return_value=mock_client),
            patch("scripts.backfill_cast_payload._fetch_cast_top15", return_value=new_cast),
        ):
            backfill(
                qdrant_url="http://localhost:6333",
                qdrant_api_key="",
                tmdb_api_key="fake_key",
                collection_name="tmdb_movies",
            )

        set_call_kwargs = mock_client.set_payload.call_args.kwargs
        payload = set_call_kwargs.get("payload", {})
        assert list(payload.keys()) == ["cast"], f"payload must only contain 'cast', got: {list(payload.keys())}"

    def test_tmdb_miss_does_not_abort_run(self) -> None:
        """A None return from _fetch_cast_top15 increments failures but continues."""
        from scripts.backfill_cast_payload import backfill

        records = [
            self._make_scroll_record(1, tmdb_id=1, cast=["A"]),
            self._make_scroll_record(2, tmdb_id=2, cast=["B"]),
        ]
        new_cast = [f"Actor {i}" for i in range(15)]

        mock_client = MagicMock()
        mock_client.scroll.return_value = (records, None)

        # First record fails (returns None), second succeeds.
        with (
            patch("scripts.backfill_cast_payload.QdrantClient", return_value=mock_client),
            patch(
                "scripts.backfill_cast_payload._fetch_cast_top15",
                side_effect=[None, new_cast],
            ),
        ):
            # Must not raise.
            backfill(
                qdrant_url="http://localhost:6333",
                qdrant_api_key="",
                tmdb_api_key="fake_key",
                collection_name="tmdb_movies",
            )

        # Only the second record was written.
        mock_client.set_payload.assert_called_once()


# ---------------------------------------------------------------------------
# AC-9 regression: collision path removed from synthesize — pre-graph gate only
# ---------------------------------------------------------------------------


class TestSuppressedCollisionSynthesisRegression:
    """Verify that the old _supplement_collision_hits mechanism is gone and that
    unique-title inform queries still work normally (0013 AC-9).

    The collision disambiguation question is now produced deterministically
    pre-graph (agent/main.py::run + build_collision_question); the LLM path
    no longer emits it.  These tests guard against that regression.
    """

    def test_supplement_collision_hits_not_in_synthesize(self) -> None:
        """_supplement_collision_hits must no longer exist in synthesize (AC-9)."""
        import agent.nodes.synthesize as syn_module

        assert not hasattr(syn_module, "_supplement_collision_hits"), (
            "_supplement_collision_hits was removed in 0013 (pre-graph gate replaces it)"
        )

    def test_synthesize_inform_node_does_not_call_find_by_exact_title(self) -> None:
        """synthesize_inform_node must not call find_by_exact_title (no supplement path).

        The collision question is now pre-graph; synthesis only answers.
        """
        from unittest.mock import patch, MagicMock

        from agent.config import AgentSettings
        from agent.nodes.synthesize import synthesize_inform_node

        rag_hits = [
            {
                "tmdb_id": 550,
                "title": "Fight Club",
                "year": 1999,
                "overview": "A soap salesman fights his alter ego.",
                "genres": ["Drama"],
                "vote_average": 8.4,
                "score": 0.95,
            }
        ]

        state = {
            "user_query": "Tell me about Fight Club",
            "rag_hits": rag_hits,
            "web_hits": [],
            "token_count": 0,
            "cost_usd": 0.0,
            "resolved_inform_tmdb_id": None,
        }

        fake_response = MagicMock()
        fake_response.content = "Fight Club (1999) is a psychological thriller directed by David Fincher."
        fake_response.usage_metadata = None
        fake_response.response_metadata = {}

        settings = AgentSettings.model_construct(
            model_agent="gpt-4o-mini",
            temperature=0.0,
        )

        with (
            patch("agent.nodes.synthesize.ChatOpenAI") as mock_llm_cls,
            patch("retrieval.movies.find_by_exact_title") as mock_scroll,
        ):
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = fake_response
            mock_llm_cls.return_value = mock_llm

            result = synthesize_inform_node(state, settings)

        # find_by_exact_title must NOT be called (supplement path removed)
        mock_scroll.assert_not_called()
        assert result["final_answer"] != ""
        assert result["recs"] == []

    def test_synthesize_inform_node_unique_title_produces_answer(self) -> None:
        """A unique-title inform query passes through synthesize_inform_node normally."""
        from unittest.mock import patch, MagicMock

        from agent.config import AgentSettings
        from agent.nodes.synthesize import synthesize_inform_node

        state = {
            "user_query": "Who directed Fight Club?",
            "rag_hits": [
                {
                    "tmdb_id": 550,
                    "title": "Fight Club",
                    "year": 1999,
                    "overview": "A soap salesman fights his alter ego.",
                    "genres": ["Drama"],
                    "vote_average": 8.4,
                    "score": 0.95,
                }
            ],
            "web_hits": [],
            "token_count": 0,
            "cost_usd": 0.0,
            "resolved_inform_tmdb_id": None,
        }

        expected_answer = "Fight Club was directed by David Fincher."
        fake_response = MagicMock()
        fake_response.content = expected_answer
        fake_response.usage_metadata = None
        fake_response.response_metadata = {}

        settings = AgentSettings.model_construct(
            model_agent="gpt-4o-mini",
            temperature=0.0,
        )

        with patch("agent.nodes.synthesize.ChatOpenAI") as mock_llm_cls:
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = fake_response
            mock_llm_cls.return_value = mock_llm

            result = synthesize_inform_node(state, settings)

        assert result["final_answer"] == expected_answer
        assert result["recs"] == []
