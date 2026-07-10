"""Unit tests for agent/tools/disambiguation.py.

All Qdrant calls are mocked — no live network required.
Covers AC-1 (detection), AC-3 (TitleCollision model), AC-4 (resolution),
AC-5 (nearest-year fuzzy match + MAX_YEAR_DISTANCE tolerance).

The canonical test case is the real "Obsession" corpus (4 films):
  tmdb_id=332672  year=2015
  tmdb_id=5155    year=1943
  tmdb_id=1339713 year=2026
  tmdb_id=4780    year=1976
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agent.tools.disambiguation import (
    MAX_YEAR_DISTANCE,
    CollisionCandidate,
    TitleCollision,
    build_collision_question,
    detect_title_collision,
    extract_title_from_query,
    resolve_year_reference,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

# The 4 real Obsession films (from live Qdrant, confirmed by the spec)
_OBSESSION_CANDIDATES = [
    CollisionCandidate(tmdb_id=332672, year=2015),
    CollisionCandidate(tmdb_id=5155, year=1943),
    CollisionCandidate(tmdb_id=1339713, year=2026),
    CollisionCandidate(tmdb_id=4780, year=1976),
]

_OBSESSION_COLLISION = TitleCollision(
    title="Obsession",
    candidates=_OBSESSION_CANDIDATES,
)


def _make_movie_hit(tmdb_id: int, year: int, title: str = "Obsession") -> MagicMock:
    """Stub a retrieval.models.MovieHit-like object."""
    hit = MagicMock()
    hit.tmdb_id = tmdb_id
    hit.year = year
    hit.title = title
    return hit


def _make_settings() -> MagicMock:
    """Stub a RetrievalSettings-like object (only needs ingestion().movies_collection)."""
    s = MagicMock()
    s.ingestion.return_value.movies_collection = "tmdb_movies"
    return s


# ---------------------------------------------------------------------------
# AC-1 + AC-3: detect_title_collision
# ---------------------------------------------------------------------------


class TestDetectTitleCollision:
    """Covers AC-1 (detection) and AC-3 (TitleCollision model shape)."""

    def test_obsession_returns_four_candidate_collision(self) -> None:
        """'When was Obsession released?' → TitleCollision with 4 candidates."""
        hits = [
            _make_movie_hit(332672, 2015),
            _make_movie_hit(5155, 1943),
            _make_movie_hit(1339713, 2026),
            _make_movie_hit(4780, 1976),
        ]
        with patch("agent.tools.disambiguation.find_by_exact_title", return_value=hits) as mock_scroll:
            result = detect_title_collision(
                "When was Obsession released?",
                settings=_make_settings(),
            )

        assert result is not None
        assert isinstance(result, TitleCollision)
        assert result.title == "Obsession"
        assert len(result.candidates) == 4
        years = {c.year for c in result.candidates}
        assert years == {1943, 1976, 2015, 2026}
        mock_scroll.assert_called_once()

    def test_single_match_title_returns_none(self) -> None:
        """A title with only one film in corpus → None (no collision)."""
        with patch(
            "agent.tools.disambiguation.find_by_exact_title",
            return_value=[_make_movie_hit(12345, 2010, title="Inception")],
        ):
            result = detect_title_collision(
                "Who directed Inception?",
                settings=_make_settings(),
            )

        assert result is None

    def test_year_pinned_query_returns_none_without_scroll(self) -> None:
        """'Obsession 2026' already has a year → None; no find_by_exact_title call."""
        with patch("agent.tools.disambiguation.find_by_exact_title") as mock_scroll:
            result = detect_title_collision(
                "Tell me about Obsession 2026",
                settings=_make_settings(),
            )

        assert result is None
        mock_scroll.assert_not_called()

    def test_no_title_extractable_returns_none_without_scroll(self) -> None:
        """A query with no extractable title → None; no scroll call."""
        with patch("agent.tools.disambiguation.find_by_exact_title") as mock_scroll:
            result = detect_title_collision(
                "recommend me something scary",
                settings=_make_settings(),
            )

        assert result is None
        mock_scroll.assert_not_called()

    def test_no_title_pronoun_query_returns_none_without_scroll(self) -> None:
        """'Tell me about it' extracts a pronoun → None (no collision lookup)."""
        with patch("agent.tools.disambiguation.find_by_exact_title") as mock_scroll:
            result = detect_title_collision(
                "Tell me about it",
                settings=_make_settings(),
            )

        assert result is None
        mock_scroll.assert_not_called()

    def test_zero_hits_returns_none(self) -> None:
        """find_by_exact_title returning [] → None."""
        with patch("agent.tools.disambiguation.find_by_exact_title", return_value=[]):
            result = detect_title_collision(
                "Who directed Xanadu",
                settings=_make_settings(),
            )

        assert result is None

    def test_collision_model_carries_tmdb_ids(self) -> None:
        """TitleCollision.candidates carries the correct tmdb_ids."""
        hits = [
            _make_movie_hit(332672, 2015),
            _make_movie_hit(4780, 1976),
        ]
        with patch("agent.tools.disambiguation.find_by_exact_title", return_value=hits):
            result = detect_title_collision(
                "Tell me about Obsession",
                settings=_make_settings(),
            )

        assert result is not None
        ids = {c.tmdb_id for c in result.candidates}
        assert ids == {332672, 4780}


# ---------------------------------------------------------------------------
# Templated question builder
# ---------------------------------------------------------------------------


class TestBuildCollisionQuestion:
    def test_obsession_question_lists_years_ascending(self) -> None:
        q = build_collision_question(_OBSESSION_COLLISION)
        assert "Obsession" in q
        assert "1943" in q
        assert "1976" in q
        assert "2015" in q
        assert "2026" in q
        # Years must appear in ascending order within the question
        idx_1943 = q.index("1943")
        idx_1976 = q.index("1976")
        idx_2015 = q.index("2015")
        idx_2026 = q.index("2026")
        assert idx_1943 < idx_1976 < idx_2015 < idx_2026

    def test_question_ends_with_prompt(self) -> None:
        q = build_collision_question(_OBSESSION_COLLISION)
        assert "which one did you mean?" in q.lower()

    def test_question_states_film_count(self) -> None:
        q = build_collision_question(_OBSESSION_COLLISION)
        assert "4" in q


# ---------------------------------------------------------------------------
# AC-4 + AC-5: resolve_year_reference
# ---------------------------------------------------------------------------


class TestResolveYearReferenceExact:
    """AC-4: exact-year resolution shapes."""

    def test_exact_year_plain(self) -> None:
        """'1976' → 1976 tmdb_id."""
        assert resolve_year_reference("1976", _OBSESSION_CANDIDATES) == 4780

    def test_exact_year_with_article(self) -> None:
        """'the 1976 one' → 1976 tmdb_id."""
        assert resolve_year_reference("the 1976 one", _OBSESSION_CANDIDATES) == 4780

    def test_exact_year_with_from(self) -> None:
        """'the one from 1976' → 1976 tmdb_id."""
        assert resolve_year_reference("the one from 1976", _OBSESSION_CANDIDATES) == 4780

    def test_exact_year_1943(self) -> None:
        assert resolve_year_reference("1943", _OBSESSION_CANDIDATES) == 5155

    def test_exact_year_2026(self) -> None:
        assert resolve_year_reference("2026", _OBSESSION_CANDIDATES) == 1339713

    def test_exact_year_2015(self) -> None:
        assert resolve_year_reference("2015", _OBSESSION_CANDIDATES) == 332672


class TestResolveYearReferenceSuperlative:
    """AC-4: superlative/relative resolution."""

    def test_newest_returns_max_year(self) -> None:
        assert resolve_year_reference("the newest one", _OBSESSION_CANDIDATES) == 1339713

    def test_latest_returns_max_year(self) -> None:
        assert resolve_year_reference("the latest", _OBSESSION_CANDIDATES) == 1339713

    def test_most_recent_returns_max_year(self) -> None:
        assert resolve_year_reference("the most recent one", _OBSESSION_CANDIDATES) == 1339713

    def test_oldest_returns_min_year(self) -> None:
        assert resolve_year_reference("the oldest one", _OBSESSION_CANDIDATES) == 5155

    def test_earliest_returns_min_year(self) -> None:
        assert resolve_year_reference("earliest version", _OBSESSION_CANDIDATES) == 5155

    def test_original_returns_min_year(self) -> None:
        assert resolve_year_reference("the original", _OBSESSION_CANDIDATES) == 5155


class TestResolveYearReferenceOrdinal:
    """AC-4: ordinal resolution (nth by ascending year)."""

    def test_first_ordinal_returns_1943(self) -> None:
        # "first" as ordinal (rank 1, ascending year = 1943)
        assert resolve_year_reference("the first one", _OBSESSION_CANDIDATES) == 5155

    def test_second_returns_1976(self) -> None:
        # Ascending: 1943, 1976, 2015, 2026 → 2nd = 1976
        assert resolve_year_reference("the second one", _OBSESSION_CANDIDATES) == 4780

    def test_third_returns_2015(self) -> None:
        assert resolve_year_reference("the third one", _OBSESSION_CANDIDATES) == 332672

    def test_fourth_returns_2026(self) -> None:
        assert resolve_year_reference("the fourth one", _OBSESSION_CANDIDATES) == 1339713

    def test_out_of_bounds_ordinal_returns_none(self) -> None:
        assert resolve_year_reference("the fifth one", _OBSESSION_CANDIDATES) is None

    def test_2nd_numeral_returns_1976(self) -> None:
        assert resolve_year_reference("the 2nd", _OBSESSION_CANDIDATES) == 4780


class TestResolveYearReferenceFuzzy:
    """AC-5: nearest-year fuzzy match with MAX_YEAR_DISTANCE tolerance.

    The headline real-transcript case: user types "the 2025 one" when offered
    years 1943 / 1976 / 2015 / 2026. Nearest is 2026 (distance 1 ≤ 1) → resolves.
    """

    def test_2025_resolves_to_2026_transcript_case(self) -> None:
        """The real-transcript edge case: 2025 → 2026 (distance 1 == MAX_YEAR_DISTANCE)."""
        result = resolve_year_reference("the 2025 one", _OBSESSION_CANDIDATES)
        assert result == 1339713  # 2026 film

    def test_1975_resolves_to_1976(self) -> None:
        """1975 → 1976 (distance 1 == MAX_YEAR_DISTANCE)."""
        result = resolve_year_reference("1975", _OBSESSION_CANDIDATES)
        assert result == 4780  # 1976 film

    def test_1990_returns_none_out_of_tolerance(self) -> None:
        """'the 1990 one' → nearest is 1976 (distance 14 > MAX_YEAR_DISTANCE=1) → None."""
        result = resolve_year_reference("the 1990 one", _OBSESSION_CANDIDATES)
        assert result is None

    def test_wildly_off_year_returns_none(self) -> None:
        """A wild-guess year (e.g. 2010) is too far from any candidate → None."""
        # Nearest is 2015 (dist 5) > MAX_YEAR_DISTANCE → None
        result = resolve_year_reference("the 2010 one", _OBSESSION_CANDIDATES)
        assert result is None

    def test_tie_resolves_to_newer(self) -> None:
        """When two candidates are equidistant, return the newer film."""
        cands = [
            CollisionCandidate(tmdb_id=1, year=2000),
            CollisionCandidate(tmdb_id=2, year=2002),
        ]
        # typed year 2001 → dist 1 from both; ties → newer (2002)
        result = resolve_year_reference("2001", cands)
        assert result == 2  # year=2002

    def test_max_year_distance_constant_is_one(self) -> None:
        """The design decision is MAX_YEAR_DISTANCE=1 (confirmed 2026-07-09)."""
        assert MAX_YEAR_DISTANCE == 1


class TestResolveYearReferenceEdgeCases:
    def test_empty_candidates_returns_none(self) -> None:
        assert resolve_year_reference("1976", []) is None

    def test_unrecognised_text_returns_none(self) -> None:
        assert resolve_year_reference("I don't know", _OBSESSION_CANDIDATES) is None

    def test_blank_answer_returns_none(self) -> None:
        assert resolve_year_reference("", _OBSESSION_CANDIDATES) is None


# ---------------------------------------------------------------------------
# extract_title_from_query — sanity coverage (the heavy tests live in
# tests/agent/test_cast_and_title_collision.py which was the original home)
# ---------------------------------------------------------------------------


class TestExtractTitleFromQuery:
    def test_inform_prefix_stripped(self) -> None:
        assert extract_title_from_query("When was Obsession released?") == "Obsession"

    def test_year_pinned_query_strips_year(self) -> None:
        # The function strips a trailing year from the extracted title
        result = extract_title_from_query("Tell me about Obsession 2026")
        # 2026 is stripped, so we get "Obsession" (or None if year-pin path fires first
        # in detect_title_collision — but here we test extract_title directly)
        assert result == "Obsession"

    def test_quoted_title_wins(self) -> None:
        assert extract_title_from_query("\"Obsession\" by Brian De Palma") == "Obsession"

    def test_recommend_query_may_return_none(self) -> None:
        # Recommend queries don't follow inform prefix patterns; extraction may
        # return something but detect_title_collision guards intent separately.
        # This just checks no crash.
        _ = extract_title_from_query("recommend me something scary")

    def test_pronoun_only_returns_none(self) -> None:
        assert extract_title_from_query("Tell me about it") is None
