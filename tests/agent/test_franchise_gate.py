"""Tests for the pre-graph franchise clarify gate in agent/main.py::run.

All TMDB, Qdrant, and LLM calls are mocked. No live network required.
Covers AC-2, AC-3, AC-6, AC-7, AC-8.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agent.config import AgentSettings
from agent.main import run
from agent.state import AgentRunResult
from agent.tools.franchise import FranchiseAmbiguity


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

KNIVES_OUT_TMDB_ID = 546554
GLASS_ONION_TMDB_ID = 764426

_AMBIGUITY = FranchiseAmbiguity(
    seed_id=KNIVES_OUT_TMDB_ID,
    seed_title="Knives Out",
    collection_name="Knives Out Collection",
    sibling_ids=[GLASS_ONION_TMDB_ID],
    question=(
        '"Knives Out" is part of the Knives Out Collection — do you want those included, '
        "or just films with a similar mystery/comedy vibe? (yes / no)"
    ),
)

_STUB_RUN_RESULT = AgentRunResult(
    final_answer="Here are some recommendations.",
    citations=[],
    tool_calls=2,
    latency_ms=100.0,
    cost_usd=0.001,
    orchestrator_turns=1,
    rag_calls=2,
    web_calls=0,
    needs_clarification=False,
    clarification_question=None,
    franchise_sibling_ids=[],
)


def _make_settings(**kwargs) -> AgentSettings:
    """Build AgentSettings with TMDB_API_KEY stubbed out."""
    return AgentSettings.model_construct(**kwargs)


# ---------------------------------------------------------------------------
# AC-2: clarify-pause short-circuits graph run
# ---------------------------------------------------------------------------

class TestClarifyPauseGate:
    def test_ambiguous_first_call_returns_needs_clarification_true(self) -> None:
        """First call with ambiguous seed → needs_clarification=True, empty citations."""
        with (
            patch("agent.main._tmdb_api_key", return_value="fake-key"),
            patch(
                "agent.main.detect_franchise_ambiguity",
                return_value=_AMBIGUITY,
            ),
            patch("agent.main.build_graph") as mock_build_graph,
        ):
            settings = AgentSettings.model_construct(
                clarification_answer=None,
                franchise_sibling_ids=[],
            )
            result = run("films like Knives Out", settings=settings)

        # Graph must NOT be invoked on a clarify-pause
        mock_build_graph.assert_not_called()

        assert result.needs_clarification is True
        assert result.clarification_question is not None
        assert "Knives Out" in result.clarification_question
        assert result.citations == []
        assert result.tool_calls == 0
        # Sibling ids forwarded so client can echo them back (AC-6)
        assert GLASS_ONION_TMDB_ID in result.franchise_sibling_ids

    def test_non_seed_query_detection_returns_none_no_pause(self) -> None:
        """When detect_franchise_ambiguity returns None (non-seed / no franchise),
        the gate does not pause and the graph runs normally (AC-8)."""
        with (
            patch("agent.main._tmdb_api_key", return_value="fake-key"),
            patch(
                "agent.main.detect_franchise_ambiguity",
                return_value=None,  # no ambiguity
            ) as mock_detect,
            patch("agent.main.build_graph") as mock_graph,
        ):
            mock_graph.return_value.invoke.return_value = {
                "final_answer": "Some recs",
                "recs": [],
                "rag_calls": 1,
                "web_calls": 0,
                "cost_usd": 0.0,
                "token_count": 0,
                "orchestrator_turns": 1,
            }
            settings = AgentSettings.model_construct(
                clarification_answer=None,
                franchise_sibling_ids=[],
                taste_profile=None,
            )
            result = run("recommend something tense", settings=settings)

        mock_detect.assert_called_once()  # detection was attempted
        assert result.needs_clarification is False
        assert mock_graph.called  # graph was invoked (no pause)

    def test_no_tmdb_key_skips_detection(self) -> None:
        """When TMDB_API_KEY is absent, detection is skipped silently (AC-8 implicit)."""
        with (
            patch("agent.main._tmdb_api_key", return_value=""),
            patch("agent.main.detect_franchise_ambiguity") as mock_detect,
            patch("agent.main.build_graph") as mock_graph,
        ):
            mock_graph.return_value.invoke.return_value = {
                "final_answer": "Some recs",
                "recs": [],
                "rag_calls": 0,
                "web_calls": 0,
                "cost_usd": 0.0,
                "token_count": 0,
                "orchestrator_turns": 1,
            }
            settings = AgentSettings.model_construct(
                clarification_answer=None,
                franchise_sibling_ids=[],
                taste_profile=None,
            )
            result = run("films like Knives Out", settings=settings)

        # Detection must not be called when no TMDB key
        mock_detect.assert_not_called()
        assert result.needs_clarification is False

    def test_detection_failure_falls_through_to_graph(self) -> None:
        """If detection raises, the run falls through to the normal graph (no crash)."""
        with (
            patch("agent.main._tmdb_api_key", return_value="fake-key"),
            patch(
                "agent.main.detect_franchise_ambiguity",
                side_effect=RuntimeError("TMDB down"),
            ),
            patch("agent.main.build_graph") as mock_graph,
        ):
            mock_graph.return_value.invoke.return_value = {
                "final_answer": "Fallback recs",
                "recs": [],
                "rag_calls": 1,
                "web_calls": 0,
                "cost_usd": 0.0,
                "token_count": 0,
                "orchestrator_turns": 1,
            }
            settings = AgentSettings.model_construct(
                clarification_answer=None,
                franchise_sibling_ids=[],
                taste_profile=None,
            )
            result = run("films like Knives Out", settings=settings)

        assert result.needs_clarification is False
        mock_graph.called


# ---------------------------------------------------------------------------
# AC-6: exclude path seeds franchise_exclude_ids into initial state
# ---------------------------------------------------------------------------

class TestResolvedAnswerExcludePath:
    def test_exclude_answer_seeds_state_with_sibling_ids(self) -> None:
        """When clarification_answer='no' and sibling_ids given, exclude ids
        reach the initial state (AC-6)."""
        captured_state: dict = {}

        def _capture_invoke(state: dict) -> dict:
            captured_state.update(state)
            return {
                "final_answer": "Recs without sequels",
                "recs": [],
                "rag_calls": 1,
                "web_calls": 0,
                "cost_usd": 0.0,
                "token_count": 0,
                "orchestrator_turns": 1,
            }

        with (
            patch("agent.main._tmdb_api_key", return_value="fake-key"),
            patch("agent.main.build_graph") as mock_graph,
        ):
            mock_graph.return_value.invoke.side_effect = _capture_invoke
            settings = AgentSettings.model_construct(
                clarification_answer="no",
                franchise_sibling_ids=[GLASS_ONION_TMDB_ID],
                taste_profile=None,
            )
            result = run("films like Knives Out", settings=settings)

        # Detection must NOT be called (single-turn hard cap, AC-7)
        # franchise_exclude_ids should be seeded from sibling_ids
        assert captured_state["franchise_exclude_ids"] == [GLASS_ONION_TMDB_ID]
        assert captured_state["franchise_include"] is False
        assert result.needs_clarification is False

    def test_include_answer_sets_franchise_include_true_empty_exclude(self) -> None:
        """When clarification_answer='yes', no exclusion filter is applied (AC-6 include path)."""
        captured_state: dict = {}

        def _capture_invoke(state: dict) -> dict:
            captured_state.update(state)
            return {
                "final_answer": "Recs with sequels",
                "recs": [],
                "rag_calls": 1,
                "web_calls": 0,
                "cost_usd": 0.0,
                "token_count": 0,
                "orchestrator_turns": 1,
            }

        with (
            patch("agent.main._tmdb_api_key", return_value="fake-key"),
            patch("agent.main.build_graph") as mock_graph,
        ):
            mock_graph.return_value.invoke.side_effect = _capture_invoke
            settings = AgentSettings.model_construct(
                clarification_answer="yes",
                franchise_sibling_ids=[GLASS_ONION_TMDB_ID],
                taste_profile=None,
            )
            result = run("films like Knives Out", settings=settings)

        assert captured_state["franchise_include"] is True
        assert captured_state["franchise_exclude_ids"] == []
        assert result.needs_clarification is False


# ---------------------------------------------------------------------------
# AC-7: unclear answer → default exclude; no infinite clarify loop
# ---------------------------------------------------------------------------

class TestUnclearAnswerFallback:
    def test_unclear_answer_defaults_to_exclude(self) -> None:
        """Unclear clarification_answer → default exclude (franchise_include=False)."""
        captured_state: dict = {}

        def _capture_invoke(state: dict) -> dict:
            captured_state.update(state)
            return {
                "final_answer": "Default recs",
                "recs": [],
                "rag_calls": 1,
                "web_calls": 0,
                "cost_usd": 0.0,
                "token_count": 0,
                "orchestrator_turns": 1,
            }

        with (
            patch("agent.main._tmdb_api_key", return_value="fake-key"),
            patch("agent.main.build_graph") as mock_graph,
        ):
            mock_graph.return_value.invoke.side_effect = _capture_invoke
            settings = AgentSettings.model_construct(
                clarification_answer="hmm I'm not sure",
                franchise_sibling_ids=[GLASS_ONION_TMDB_ID],
                taste_profile=None,
            )
            result = run("films like Knives Out", settings=settings)

        # Unclear → default exclude
        assert captured_state["franchise_include"] is False
        assert captured_state["franchise_exclude_ids"] == [GLASS_ONION_TMDB_ID]
        assert result.needs_clarification is False

    def test_second_call_with_answer_does_not_rerun_detection(self) -> None:
        """When clarification_answer is present, detect_franchise_ambiguity is
        never called (single clarify turn hard cap, AC-7)."""
        with (
            patch("agent.main._tmdb_api_key", return_value="fake-key"),
            patch("agent.main.detect_franchise_ambiguity") as mock_detect,
            patch("agent.main.build_graph") as mock_graph,
        ):
            mock_graph.return_value.invoke.return_value = {
                "final_answer": "Recs",
                "recs": [],
                "rag_calls": 1,
                "web_calls": 0,
                "cost_usd": 0.0,
                "token_count": 0,
                "orchestrator_turns": 1,
            }
            settings = AgentSettings.model_construct(
                clarification_answer="no",
                franchise_sibling_ids=[GLASS_ONION_TMDB_ID],
                taste_profile=None,
            )
            run("films like Knives Out", settings=settings)

        mock_detect.assert_not_called()  # hard cap: no detection on second call


# ---------------------------------------------------------------------------
# AC-3: non-franchise query regression — byte-identical behaviour
# ---------------------------------------------------------------------------

class TestNonFranchiseRegression:
    def test_non_franchise_query_needs_clarification_false(self) -> None:
        """A non-franchise query (detection returns None) behaves exactly as before."""
        with (
            patch("agent.main._tmdb_api_key", return_value="fake-key"),
            patch(
                "agent.main.detect_franchise_ambiguity",
                return_value=None,
            ),
            patch("agent.main.build_graph") as mock_graph,
        ):
            mock_graph.return_value.invoke.return_value = {
                "final_answer": "A slow film: Paris, Texas.",
                "recs": [{"tmdb_id": 655, "title": "Paris, Texas", "year": 1984, "why_for_you": "quiet"}],
                "rag_calls": 2,
                "web_calls": 0,
                "cost_usd": 0.0012,
                "token_count": 50,
                "orchestrator_turns": 1,
            }
            settings = AgentSettings.model_construct(
                clarification_answer=None,
                franchise_sibling_ids=[],
                taste_profile=None,
            )
            result = run("recommend something slow and meditative", settings=settings)

        assert result.needs_clarification is False
        assert result.clarification_question is None
        assert len(result.citations) == 1
        assert result.citations[0].tmdb_id == 655
