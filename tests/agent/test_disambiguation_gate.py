"""Tests for the pre-graph collision disambiguation gate in agent/main.py::run.

All Qdrant, TMDB, and LLM calls are mocked.  No live network required.
Covers AC-2 (clarify-pause short-circuits graph), AC-6 (resolved single-film
seed), AC-7 (unresolvable → newest fallback + one-turn cap), AC-8 (scope).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agent.config import AgentSettings
from agent.main import run
from agent.state import AgentRunResult
from agent.tools.disambiguation import CollisionCandidate, TitleCollision


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# Four Obsession candidates (real live data)
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


def _make_settings(**kwargs) -> AgentSettings:
    return AgentSettings.model_construct(**kwargs)


def _stub_graph_result(answer: str = "Obsession (1976) is a De Palma thriller.") -> dict:
    return {
        "final_answer": answer,
        "recs": [],
        "rag_calls": 1,
        "web_calls": 0,
        "cost_usd": 0.001,
        "token_count": 50,
        "orchestrator_turns": 1,
    }


# ---------------------------------------------------------------------------
# AC-2: collision-pause short-circuits graph (no graph.invoke on first call)
# ---------------------------------------------------------------------------


class TestCollisionClarifyPause:
    def test_first_call_returns_needs_clarification_true(self) -> None:
        """Colliding-title inform query, first call → pause with templated question."""
        with (
            patch(
                "agent.main.detect_title_collision",
                return_value=_OBSESSION_COLLISION,
            ),
            patch("agent.main.build_graph") as mock_graph,
            patch("agent.main._tmdb_api_key", return_value="fake-key"),
        ):
            settings = _make_settings(
                clarification_answer=None,
                franchise_sibling_ids=[],
                taste_profile=None,
            )
            result = run("When was Obsession released?", settings=settings)

        # Graph must NOT be invoked on a clarify-pause (AC-2)
        mock_graph.assert_not_called()

        assert result.needs_clarification is True
        assert result.clarification_question is not None
        assert "Obsession" in result.clarification_question
        # All 4 candidate years must appear in the question
        for year in ("1943", "1976", "2015", "2026"):
            assert year in result.clarification_question
        assert result.citations == []
        assert result.tool_calls == 0
        assert result.cost_usd == 0.0
        assert result.final_answer == result.clarification_question

    def test_unique_title_no_pause_graph_runs(self) -> None:
        """A title with no collision (detect returns None) falls through to graph."""
        with (
            patch(
                "agent.main.detect_title_collision",
                return_value=None,
            ) as mock_detect,
            patch("agent.main.detect_franchise_ambiguity", return_value=None),
            patch("agent.main._tmdb_api_key", return_value="fake-key"),
            patch("agent.main.build_graph") as mock_graph,
        ):
            mock_graph.return_value.invoke.return_value = _stub_graph_result("Inception is a 2010 film.")
            settings = _make_settings(
                clarification_answer=None,
                franchise_sibling_ids=[],
                taste_profile=None,
            )
            result = run("Tell me about Inception", settings=settings)

        mock_detect.assert_called_once()
        assert result.needs_clarification is False
        assert mock_graph.called

    def test_recommend_query_collision_not_triggered(self) -> None:
        """A recommend query does not trigger the collision gate (AC-8 scope guard)."""
        # detect_title_collision returns None for recommend queries because
        # extract_title_from_query strips "recommend" and returns None or a non-title.
        # We verify the gate behaves correctly by patching detect to return None.
        with (
            patch(
                "agent.main.detect_title_collision",
                return_value=None,
            ) as mock_detect,
            patch("agent.main.detect_franchise_ambiguity", return_value=None),
            patch("agent.main._tmdb_api_key", return_value="fake-key"),
            patch("agent.main.build_graph") as mock_graph,
        ):
            mock_graph.return_value.invoke.return_value = _stub_graph_result("Here are some recs.")
            settings = _make_settings(
                clarification_answer=None,
                franchise_sibling_ids=[],
                taste_profile=None,
            )
            result = run("recommend something slow", settings=settings)

        assert result.needs_clarification is False
        mock_detect.assert_called_once()


# ---------------------------------------------------------------------------
# AC-6: second call resolves year, seeds resolved_inform_tmdb_id
# ---------------------------------------------------------------------------


class TestCollisionResolutionSecondCall:
    def test_1976_answer_seeds_resolved_tmdb_id(self) -> None:
        """Second call 'the 1976 one' → resolved_inform_tmdb_id = 4780 (1976 Obsession)."""
        captured_state: dict = {}

        def _capture_invoke(state: dict) -> dict:
            captured_state.update(state)
            return _stub_graph_result()

        with (
            patch(
                "agent.main.detect_title_collision",
                return_value=_OBSESSION_COLLISION,
            ),
            patch("agent.main.build_graph") as mock_graph,
        ):
            mock_graph.return_value.invoke.side_effect = _capture_invoke
            settings = _make_settings(
                clarification_answer="the 1976 one",
                franchise_sibling_ids=[],
                taste_profile=None,
            )
            result = run("When was Obsession released?", settings=settings)

        # Graph IS invoked on the second call
        mock_graph.assert_called_once()
        assert captured_state["resolved_inform_tmdb_id"] == 4780  # 1976 film
        assert result.needs_clarification is False

    def test_newest_answer_seeds_2026_film(self) -> None:
        """'the newest one' resolves to the 2026 Obsession (tmdb_id=1339713)."""
        captured_state: dict = {}

        def _capture_invoke(state: dict) -> dict:
            captured_state.update(state)
            return _stub_graph_result()

        with (
            patch(
                "agent.main.detect_title_collision",
                return_value=_OBSESSION_COLLISION,
            ),
            patch("agent.main.build_graph") as mock_graph,
        ):
            mock_graph.return_value.invoke.side_effect = _capture_invoke
            settings = _make_settings(
                clarification_answer="the newest one",
                franchise_sibling_ids=[],
                taste_profile=None,
            )
            run("When was Obsession released?", settings=settings)

        assert captured_state["resolved_inform_tmdb_id"] == 1339713

    def test_2025_answer_fuzzy_matches_2026(self) -> None:
        """'the 2025 one' (real transcript case) → resolves to 2026 film (dist=1)."""
        captured_state: dict = {}

        def _capture_invoke(state: dict) -> dict:
            captured_state.update(state)
            return _stub_graph_result()

        with (
            patch(
                "agent.main.detect_title_collision",
                return_value=_OBSESSION_COLLISION,
            ),
            patch("agent.main.build_graph") as mock_graph,
        ):
            mock_graph.return_value.invoke.side_effect = _capture_invoke
            settings = _make_settings(
                clarification_answer="the 2025 one",
                franchise_sibling_ids=[],
                taste_profile=None,
            )
            run("When was Obsession released?", settings=settings)

        assert captured_state["resolved_inform_tmdb_id"] == 1339713  # 2026 film

    def test_second_call_result_is_not_clarify_pause(self) -> None:
        """Second call with any answer must not re-ask (no re-pause)."""
        with (
            patch(
                "agent.main.detect_title_collision",
                return_value=_OBSESSION_COLLISION,
            ),
            patch("agent.main.build_graph") as mock_graph,
        ):
            mock_graph.return_value.invoke.return_value = _stub_graph_result()
            settings = _make_settings(
                clarification_answer="1976",
                franchise_sibling_ids=[],
                taste_profile=None,
            )
            result = run("When was Obsession released?", settings=settings)

        assert result.needs_clarification is False
        assert result.clarification_question is None


# ---------------------------------------------------------------------------
# AC-7: unresolvable answer → newest-candidate fallback; no infinite loop
# ---------------------------------------------------------------------------


class TestUnresolvableAnswerFallback:
    def test_out_of_tolerance_year_defaults_to_newest(self) -> None:
        """'the 1990 one' (out-of-tolerance) → newest fallback (tmdb_id=1339713, 2026)."""
        captured_state: dict = {}

        def _capture_invoke(state: dict) -> dict:
            captured_state.update(state)
            return _stub_graph_result()

        with (
            patch(
                "agent.main.detect_title_collision",
                return_value=_OBSESSION_COLLISION,
            ),
            patch("agent.main.build_graph") as mock_graph,
        ):
            mock_graph.return_value.invoke.side_effect = _capture_invoke
            settings = _make_settings(
                clarification_answer="the 1990 one",
                franchise_sibling_ids=[],
                taste_profile=None,
            )
            result = run("When was Obsession released?", settings=settings)

        # Out-of-tolerance → AC-7 default: newest candidate (2026)
        assert captured_state["resolved_inform_tmdb_id"] == 1339713
        assert result.needs_clarification is False

    def test_gibberish_answer_defaults_to_newest(self) -> None:
        """Unrecognisable answer → newest fallback."""
        captured_state: dict = {}

        def _capture_invoke(state: dict) -> dict:
            captured_state.update(state)
            return _stub_graph_result()

        with (
            patch(
                "agent.main.detect_title_collision",
                return_value=_OBSESSION_COLLISION,
            ),
            patch("agent.main.build_graph") as mock_graph,
        ):
            mock_graph.return_value.invoke.side_effect = _capture_invoke
            settings = _make_settings(
                clarification_answer="umm idk",
                franchise_sibling_ids=[],
                taste_profile=None,
            )
            run("When was Obsession released?", settings=settings)

        assert captured_state["resolved_inform_tmdb_id"] == 1339713  # 2026, newest

    def test_second_call_never_re_pauses(self) -> None:
        """A request with clarification_answer never returns needs_clarification=True."""
        with (
            patch(
                "agent.main.detect_title_collision",
                return_value=_OBSESSION_COLLISION,
            ),
            patch("agent.main.build_graph") as mock_graph,
        ):
            mock_graph.return_value.invoke.return_value = _stub_graph_result()
            settings = _make_settings(
                clarification_answer="the 1990 one",  # out-of-tolerance, uses fallback
                franchise_sibling_ids=[],
                taste_profile=None,
            )
            result = run("When was Obsession released?", settings=settings)

        assert result.needs_clarification is False

    def test_detection_failure_falls_through_to_graph(self) -> None:
        """If collision detection raises on first call, falls through to graph (no crash)."""
        with (
            patch(
                "agent.main.detect_title_collision",
                side_effect=RuntimeError("Qdrant down"),
            ),
            patch("agent.main.detect_franchise_ambiguity", return_value=None),
            patch("agent.main._tmdb_api_key", return_value="fake-key"),
            patch("agent.main.build_graph") as mock_graph,
        ):
            mock_graph.return_value.invoke.return_value = _stub_graph_result()
            settings = _make_settings(
                clarification_answer=None,
                franchise_sibling_ids=[],
                taste_profile=None,
            )
            result = run("Tell me about Obsession", settings=settings)

        assert result.needs_clarification is False
        assert mock_graph.called


# ---------------------------------------------------------------------------
# AC-8: scope — franchise path unaffected
# ---------------------------------------------------------------------------


class TestFranchiseGateUnaffected:
    def test_franchise_second_call_resolves_include_exclude_correctly(self) -> None:
        """When no collision but clarification_answer present, franchise resolve runs."""
        captured_state: dict = {}

        def _capture_invoke(state: dict) -> dict:
            captured_state.update(state)
            return _stub_graph_result("Films like Knives Out.")

        # detect_title_collision returns None (no collision for this query)
        with (
            patch(
                "agent.main.detect_title_collision",
                return_value=None,
            ),
            patch("agent.main.build_graph") as mock_graph,
        ):
            mock_graph.return_value.invoke.side_effect = _capture_invoke
            settings = _make_settings(
                clarification_answer="no",
                franchise_sibling_ids=[764426],  # Glass Onion
                taste_profile=None,
            )
            result = run("films like Knives Out", settings=settings)

        # Franchise exclude path: sibling ids seeded into state
        assert captured_state["franchise_exclude_ids"] == [764426]
        assert captured_state["franchise_include"] is False
        # No collision → resolved_inform_tmdb_id is None
        assert captured_state["resolved_inform_tmdb_id"] is None
        assert result.needs_clarification is False
