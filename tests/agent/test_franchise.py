"""Unit tests for agent/tools/franchise.py — franchise ambiguity detection.

All TMDB and Qdrant calls are mocked. No live network required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agent.tools.franchise import (
    FranchiseAmbiguity,
    _point_id,
    detect_franchise_ambiguity,
    resolve_clarification,
)


# ---------------------------------------------------------------------------
# Helpers / shared fixtures
# ---------------------------------------------------------------------------

KNIVES_OUT_TMDB_ID = 546554
GLASS_ONION_TMDB_ID = 764426
WAKE_UP_DEAD_MAN_TMDB_ID = 1072790  # hypothetical third entry

KNIVES_OUT_COLLECTION_ID = 784720
KNIVES_OUT_COLLECTION_NAME = "Knives Out Collection"

_KNIVES_OUT_DETAILS = {
    "id": KNIVES_OUT_TMDB_ID,
    "title": "Knives Out",
    "genres": [{"id": 9648, "name": "Mystery"}, {"id": 35, "name": "Comedy"}],
    "belongs_to_collection": {
        "id": KNIVES_OUT_COLLECTION_ID,
        "name": KNIVES_OUT_COLLECTION_NAME,
        "poster_path": "/abc.jpg",
        "backdrop_path": "/def.jpg",
    },
}

_KNIVES_OUT_COLLECTION = {
    "id": KNIVES_OUT_COLLECTION_ID,
    "name": KNIVES_OUT_COLLECTION_NAME,
    "parts": [
        {"id": KNIVES_OUT_TMDB_ID, "title": "Knives Out"},
        {"id": GLASS_ONION_TMDB_ID, "title": "Glass Onion: A Knives Out Mystery"},
        {"id": WAKE_UP_DEAD_MAN_TMDB_ID, "title": "Wake Up Dead Man: A Knives Out Mystery"},
    ],
}


def _make_qdrant_record(tmdb_id: int) -> MagicMock:
    rec = MagicMock()
    rec.id = _point_id(tmdb_id)
    return rec


# ---------------------------------------------------------------------------
# Tests: detect_franchise_ambiguity
# ---------------------------------------------------------------------------

class TestDetectFranchiseAmbiguity:
    def test_non_seed_query_returns_none_no_tmdb_call(self) -> None:
        """A generic / non-seed-shaped query must return None with zero TMDB calls."""
        with (
            patch("agent.tools.franchise.extract_seed_title", return_value=None) as mock_seed,
            patch("agent.tools.franchise._fetch_movie_details") as mock_fetch,
        ):
            result = detect_franchise_ambiguity(
                "recommend something tense and slow", tmdb_api_key="fake"
            )

        assert result is None
        mock_seed.assert_called_once()
        mock_fetch.assert_not_called()  # AC-8: zero TMDB calls for non-seed queries

    def test_knives_out_returns_ambiguity_with_sibling_ids(self) -> None:
        """'Films like Knives Out' → FranchiseAmbiguity with corpus sibling ids."""
        sibling_record_glass_onion = _make_qdrant_record(GLASS_ONION_TMDB_ID)
        sibling_record_wake_up = _make_qdrant_record(WAKE_UP_DEAD_MAN_TMDB_ID)

        with (
            patch("agent.tools.franchise.search_tmdb", return_value=KNIVES_OUT_TMDB_ID),
            patch(
                "agent.tools.franchise._fetch_movie_details",
                return_value=_KNIVES_OUT_DETAILS,
            ),
            patch(
                "agent.tools.franchise._fetch_collection_members",
                return_value=[
                    KNIVES_OUT_TMDB_ID,
                    GLASS_ONION_TMDB_ID,
                    WAKE_UP_DEAD_MAN_TMDB_ID,
                ],
            ),
            patch(
                "agent.tools.franchise.get_qdrant_client"
            ) as mock_get_client,
        ):
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client
            mock_client.retrieve.return_value = [
                sibling_record_glass_onion,
                sibling_record_wake_up,
            ]

            result = detect_franchise_ambiguity(
                "films like Knives Out", tmdb_api_key="fake"
            )

        assert result is not None
        assert isinstance(result, FranchiseAmbiguity)
        assert result.seed_id == KNIVES_OUT_TMDB_ID
        assert result.seed_title == "Knives Out"
        assert result.collection_name == KNIVES_OUT_COLLECTION_NAME
        # Siblings should exclude the seed itself
        assert KNIVES_OUT_TMDB_ID not in result.sibling_ids
        assert GLASS_ONION_TMDB_ID in result.sibling_ids
        assert WAKE_UP_DEAD_MAN_TMDB_ID in result.sibling_ids
        assert result.question  # must be non-empty
        assert "Knives Out" in result.question

    def test_standalone_film_no_collection_returns_none(self) -> None:
        """A film with no belongs_to_collection returns None."""
        details_no_collection = {
            "id": 12345,
            "title": "Some Standalone Film",
            "genres": [{"id": 18, "name": "Drama"}],
            "belongs_to_collection": None,
        }
        with (
            patch("agent.tools.franchise.search_tmdb", return_value=12345),
            patch(
                "agent.tools.franchise._fetch_movie_details",
                return_value=details_no_collection,
            ),
        ):
            result = detect_franchise_ambiguity(
                "films like Some Standalone Film", tmdb_api_key="fake"
            )

        assert result is None

    def test_collection_with_no_corpus_siblings_returns_none(self) -> None:
        """When the seed is the only corpus member of its collection → None."""
        with (
            patch("agent.tools.franchise.search_tmdb", return_value=KNIVES_OUT_TMDB_ID),
            patch(
                "agent.tools.franchise._fetch_movie_details",
                return_value=_KNIVES_OUT_DETAILS,
            ),
            patch(
                "agent.tools.franchise._fetch_collection_members",
                return_value=[KNIVES_OUT_TMDB_ID, GLASS_ONION_TMDB_ID],
            ),
            patch("agent.tools.franchise.get_qdrant_client") as mock_get_client,
        ):
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client
            # Only the seed is returned from Qdrant (siblings not in corpus)
            mock_client.retrieve.return_value = []  # no siblings found

            result = detect_franchise_ambiguity(
                "films like Knives Out", tmdb_api_key="fake"
            )

        assert result is None

    def test_seed_not_found_in_tmdb_returns_none(self) -> None:
        """When TMDB search fails to resolve the seed → None (no further calls)."""
        with (
            patch("agent.tools.franchise.search_tmdb", return_value=None),
            patch("agent.tools.franchise._fetch_movie_details") as mock_fetch,
        ):
            result = detect_franchise_ambiguity(
                "films like UnknownFilmXYZ", tmdb_api_key="fake"
            )

        assert result is None
        mock_fetch.assert_not_called()

    def test_question_contains_genre_words(self) -> None:
        """When genres are present, the clarification question mentions them."""
        sibling_record = _make_qdrant_record(GLASS_ONION_TMDB_ID)

        with (
            patch("agent.tools.franchise.search_tmdb", return_value=KNIVES_OUT_TMDB_ID),
            patch(
                "agent.tools.franchise._fetch_movie_details",
                return_value=_KNIVES_OUT_DETAILS,
            ),
            patch(
                "agent.tools.franchise._fetch_collection_members",
                return_value=[KNIVES_OUT_TMDB_ID, GLASS_ONION_TMDB_ID],
            ),
            patch("agent.tools.franchise.get_qdrant_client") as mock_get_client,
        ):
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client
            mock_client.retrieve.return_value = [sibling_record]

            result = detect_franchise_ambiguity(
                "films like Knives Out", tmdb_api_key="fake"
            )

        assert result is not None
        # The question should incorporate genre words (mystery/comedy)
        assert "mystery" in result.question.lower() or "comedy" in result.question.lower()


# ---------------------------------------------------------------------------
# Tests: resolve_clarification
# ---------------------------------------------------------------------------

class TestResolveClarification:
    @pytest.mark.parametrize("answer", [
        "yes",
        "Yes",
        "YES",
        "yes please",
        "include them",
        "include sequels",
        "sure, include them",
        "yeah include those",
        "absolutely",
        "sequels too",
        "ok sure",
        "y",
    ])
    def test_affirmative_answers_return_true(self, answer: str) -> None:
        assert resolve_clarification(answer) is True

    @pytest.mark.parametrize("answer", [
        "no",
        "No",
        "NO",
        "just the vibe",
        "only similar films",
        "nope",
        "exclude them",
        "skip the sequels",
        "nah just the vibe",
        "n",
        "avoid sequels",
    ])
    def test_negative_answers_return_false(self, answer: str) -> None:
        assert resolve_clarification(answer) is False

    @pytest.mark.parametrize("answer", [
        "hmm I'm not sure",
        "what do you mean",
        "tell me more",
        "maybe",
        "it depends",
        "",
        "   ",
    ])
    def test_unclear_answers_return_none(self, answer: str) -> None:
        assert resolve_clarification(answer) is None

    def test_conflicting_tokens_return_none(self) -> None:
        """When both affirmative and negative tokens appear, result is unclear."""
        result = resolve_clarification("yes but no")
        assert result is None
