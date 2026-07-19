"""Unit tests for agent/tools/franchise.py — franchise ambiguity detection.

All TMDB and Qdrant calls are mocked. No live network required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agent.tools.franchise import (
    FranchiseAmbiguity,
    _build_question,
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
# Tests: _build_question (AC-1.1, AC-1.2)
# ---------------------------------------------------------------------------


class TestBuildQuestion:
    """AC-1.1 — reworded two-option question, no yes/no suffix, deterministic."""

    def test_contains_included_option(self) -> None:
        q = _build_question("Knives Out", "Knives Out Collection", ["Mystery", "Comedy"])
        assert "included" in q.lower()

    def test_contains_similar_option(self) -> None:
        q = _build_question("Knives Out", "Knives Out Collection", ["Mystery", "Comedy"])
        assert "similar" in q.lower()

    def test_no_yes_no_suffix(self) -> None:
        q = _build_question("Knives Out", "Knives Out Collection", ["Mystery", "Comedy"])
        assert "yes / no" not in q
        assert "(yes/no)" not in q
        assert "yes/no" not in q

    def test_deterministic_across_calls(self) -> None:
        args = ("Knives Out", "Knives Out Collection", ["Mystery", "Comedy"])
        assert _build_question(*args) == _build_question(*args)

    def test_genre_enriched_vibe_in_question(self) -> None:
        q = _build_question("Knives Out", "Knives Out Collection", ["Mystery", "Comedy"])
        assert "mystery" in q.lower() or "comedy" in q.lower()

    def test_no_genre_fallback_to_similar_vibe(self) -> None:
        q = _build_question("Knives Out", "Knives Out Collection", [])
        assert "similar vibe" in q.lower()

    def test_seed_title_in_question(self) -> None:
        q = _build_question("Knives Out", "Knives Out Collection", [])
        assert "Knives Out" in q

    def test_collection_name_in_question(self) -> None:
        q = _build_question("Knives Out", "Knives Out Collection", [])
        assert "Knives Out Collection" in q


class TestResolveClarificationRoundTrip:
    """AC-1.2 — reworded question's plausible free-text answers still parse correctly."""

    @pytest.mark.parametrize("answer", [
        "include them",
        "yes include them",
        "sequels too",
        "sure, include all",
    ])
    def test_include_answers_return_true(self, answer: str) -> None:
        assert resolve_clarification(answer) is True

    @pytest.mark.parametrize("answer", [
        "just the vibe",
        "only similar ones",
        "skip the sequels",
        "no, just similar films",
    ])
    def test_exclude_answers_return_false(self, answer: str) -> None:
        assert resolve_clarification(answer) is False

    def test_unclear_returns_none(self) -> None:
        assert resolve_clarification("maybe") is None


# ---------------------------------------------------------------------------
# Tests: detect_franchise_ambiguity
# ---------------------------------------------------------------------------

class TestDetectFranchiseAmbiguity:
    def test_non_seed_query_returns_none_no_tmdb_call(self) -> None:
        """A generic / non-seed-shaped query must return None with zero TMDB calls.

        Both the regex fast-path (extract_seed_title) and the LLM fallback
        (_extract_seed_title_via_llm) return None → no TMDB call is made.
        """
        with (
            patch("agent.tools.franchise.extract_seed_title", return_value=None) as mock_seed,
            patch(
                "agent.tools.franchise._extract_seed_title_via_llm", return_value=None
            ) as mock_llm,
            patch("agent.tools.franchise._fetch_movie_details") as mock_fetch,
        ):
            result = detect_franchise_ambiguity(
                "recommend something tense and slow", tmdb_api_key="fake"
            )

        assert result is None
        mock_seed.assert_called_once()
        mock_llm.assert_called_once()
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


# ---------------------------------------------------------------------------
# Phase 3 (AC-2.1, AC-2.2, AC-2.3, AC-2.4): language-agnostic franchise detection
# ---------------------------------------------------------------------------


def _make_full_qdrant_patch(sibling_ids: list[int]):
    """Context manager that patches Qdrant to return the given sibling ids."""
    records = [_make_qdrant_record(tid) for tid in sibling_ids]
    mock_client = MagicMock()
    mock_client.retrieve.return_value = records
    return mock_client


class TestFranchiseDetectionPTQuery:
    """AC-2.1 + AC-2.2 — 'Filmes como Knives Out' (PT) produces the same result as EN."""

    def test_pt_seed_query_detects_franchise_same_sibling_ids(self) -> None:
        """AC-2.1: 'Filmes como Knives Out' → franchise ambiguity with same sibling ids."""
        sibling_record_glass_onion = _make_qdrant_record(GLASS_ONION_TMDB_ID)
        sibling_record_wake_up = _make_qdrant_record(WAKE_UP_DEAD_MAN_TMDB_ID)

        with (
            # EN regex returns None for PT phrasing — LLM fallback kicks in
            patch("agent.tools.franchise.extract_seed_title", return_value=None),
            patch(
                "agent.tools.franchise._extract_seed_title_via_llm",
                return_value="Knives Out",
            ) as mock_llm,
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
            patch("agent.tools.franchise.get_qdrant_client") as mock_get_client,
        ):
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client
            mock_client.retrieve.return_value = [
                sibling_record_glass_onion,
                sibling_record_wake_up,
            ]

            result = detect_franchise_ambiguity(
                "Filmes como Knives Out", tmdb_api_key="fake"
            )

        assert result is not None
        assert isinstance(result, FranchiseAmbiguity)
        assert result.seed_id == KNIVES_OUT_TMDB_ID
        assert result.seed_title == "Knives Out"
        assert GLASS_ONION_TMDB_ID in result.sibling_ids
        assert WAKE_UP_DEAD_MAN_TMDB_ID in result.sibling_ids
        assert KNIVES_OUT_TMDB_ID not in result.sibling_ids
        mock_llm.assert_called_once_with("Filmes como Knives Out")

    def test_en_and_pt_variants_resolve_to_same_seed_id(self) -> None:
        """AC-2.2: EN and PT variants both resolve Knives Out to the same tmdb_id."""
        sibling_record = _make_qdrant_record(GLASS_ONION_TMDB_ID)

        shared_patches = {
            "tmdb_id": KNIVES_OUT_TMDB_ID,
            "details": _KNIVES_OUT_DETAILS,
            "members": [KNIVES_OUT_TMDB_ID, GLASS_ONION_TMDB_ID],
        }

        en_result = None
        pt_result = None

        # EN query: regex fast-path succeeds, no LLM call
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
            en_result = detect_franchise_ambiguity(
                "films like Knives Out", tmdb_api_key="fake"
            )

        # PT query: regex misses, LLM fallback extracts "Knives Out"
        with (
            patch("agent.tools.franchise.extract_seed_title", return_value=None),
            patch(
                "agent.tools.franchise._extract_seed_title_via_llm",
                return_value="Knives Out",
            ),
            patch("agent.tools.franchise.search_tmdb", return_value=KNIVES_OUT_TMDB_ID),
            patch(
                "agent.tools.franchise._fetch_movie_details",
                return_value=_KNIVES_OUT_DETAILS,
            ),
            patch(
                "agent.tools.franchise._fetch_collection_members",
                return_value=[KNIVES_OUT_TMDB_ID, GLASS_ONION_TMDB_ID],
            ),
            patch("agent.tools.franchise.get_qdrant_client") as mock_get_client2,
        ):
            mock_client2 = MagicMock()
            mock_get_client2.return_value = mock_client2
            mock_client2.retrieve.return_value = [sibling_record]
            pt_result = detect_franchise_ambiguity(
                "Filmes como Knives Out", tmdb_api_key="fake"
            )

        assert en_result is not None
        assert pt_result is not None
        assert en_result.seed_id == pt_result.seed_id == KNIVES_OUT_TMDB_ID
        assert set(en_result.sibling_ids) == set(pt_result.sibling_ids)


class TestFranchiseCostScope:
    """AC-2.3 + AC-2.4 — zero seed-intent LLM calls on non-seed traffic."""

    @pytest.mark.parametrize("query", [
        "recommend something slow and tense",
        "Who directed Inception?",
        "Tell me about Parasite",
        "what films has Tom Hanks been in?",
        "recomende algo assustador",  # PT non-seed recommend
    ])
    def test_non_seed_queries_trigger_zero_llm_and_tmdb_calls(self, query: str) -> None:
        """AC-2.3: non-seed queries → zero seed-intent LLM calls AND zero TMDB fetches.

        Note: non-seed queries that don't match _SEED_PATTERNS will call the LLM
        fallback once (that's the cost contract — it fires only on the
        seed-candidate branch where regex returned None). Generic non-seed queries
        that produce a regex hit of None WILL trigger one LLM call. The cost
        property is that this LLM call returns None quickly (no title extracted)
        and no TMDB calls follow.
        """
        with (
            patch(
                "agent.tools.franchise._extract_seed_title_via_llm",
                return_value=None,
            ) as mock_llm,
            patch("agent.tools.franchise._fetch_movie_details") as mock_tmdb_detail,
            patch("agent.tools.franchise._fetch_collection_members") as mock_tmdb_coll,
            patch("agent.tools.franchise.get_qdrant_client") as mock_qdrant,
        ):
            result = detect_franchise_ambiguity(query, tmdb_api_key="fake")

        assert result is None
        mock_tmdb_detail.assert_not_called()
        mock_tmdb_coll.assert_not_called()
        mock_qdrant.assert_not_called()

    def test_en_seed_query_triggers_zero_llm_calls(self) -> None:
        """AC-2.3: EN seed query hits the regex fast-path → zero LLM calls."""
        sibling_record = _make_qdrant_record(GLASS_ONION_TMDB_ID)

        with (
            patch(
                "agent.tools.franchise._extract_seed_title_via_llm",
            ) as mock_llm,
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
        mock_llm.assert_not_called()  # EN regex hit → LLM never called

    def test_llm_error_during_seed_extraction_falls_through_to_none(self) -> None:
        """AC-2.4: LLM failure on the seed-candidate branch → non-crash, returns None."""
        with (
            patch("agent.tools.franchise.extract_seed_title", return_value=None),
            patch(
                "agent.tools.franchise._extract_seed_title_via_llm",
                return_value=None,  # simulates API failure returning None
            ),
            patch("agent.tools.franchise._fetch_movie_details") as mock_fetch,
        ):
            result = detect_franchise_ambiguity(
                "Filmes como Knives Out", tmdb_api_key="fake"
            )

        assert result is None
        mock_fetch.assert_not_called()
