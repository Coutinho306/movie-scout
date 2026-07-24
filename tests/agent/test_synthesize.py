"""Unit tests for the low-signal output gate in agent/nodes/synthesize.py.

All LLM calls are mocked. No network required.
Covers AC-3 (floor gate), AC-4 (Sinval troncho / golden query E2E), AC-5 (prompt delimiters).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agent.config import AgentSettings
from agent.nodes.synthesize import (
    SCORE_FLOOR,
    _DEFLECTION_ANSWER,
    _hits_are_rrf_mode,
    _is_rrf_score,
    synthesize_node,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(
    rag_hits: list[dict],
    user_query: str = "recommend a thriller",
    *,
    web_hits: list[dict] | None = None,
) -> dict:
    return {
        "user_query": user_query,
        "rag_hits": rag_hits,
        "web_hits": web_hits or [],
        "token_count": 0,
        "cost_usd": 0.0,
    }


def _make_hit(tmdb_id: int, score: float) -> dict:
    return {
        "tmdb_id": tmdb_id,
        "title": f"Film {tmdb_id}",
        "score": score,
        "overview": "A film.",
        "genres": ["Drama"],
    }


def _make_rrf_hit(tmdb_id: int, rank: int) -> dict:
    """Simulate an RRF-mode hit (score = 1/rank)."""
    return {
        "tmdb_id": tmdb_id,
        "title": f"Film {tmdb_id}",
        "score": 1.0 / rank,
        "overview": "A film.",
        "genres": ["Drama"],
    }


def _stub_llm_response(recs: list[dict]) -> MagicMock:
    """Return a MagicMock response + parser that emits `recs`."""
    import json
    response = MagicMock()
    response.usage_metadata = {"input_tokens": 10, "output_tokens": 20}
    response.response_metadata = {}
    return response, json.dumps(recs)


def _settings() -> AgentSettings:
    return AgentSettings.model_construct()


# ---------------------------------------------------------------------------
# Internal helper tests
# ---------------------------------------------------------------------------

class TestIsRrfScore:
    def test_rank1(self) -> None:
        assert _is_rrf_score(1.0) is True

    def test_rank2(self) -> None:
        assert _is_rrf_score(0.5) is True

    def test_rank3(self) -> None:
        assert _is_rrf_score(1 / 3) is True

    def test_dense_score(self) -> None:
        assert _is_rrf_score(0.52) is False

    def test_zero(self) -> None:
        assert _is_rrf_score(0.0) is False


class TestHitsAreRrfMode:
    def test_empty_hits(self) -> None:
        assert _hits_are_rrf_mode([]) is False

    def test_all_rrf(self) -> None:
        hits = [_make_rrf_hit(1, 1), _make_rrf_hit(2, 2), _make_rrf_hit(3, 3)]
        assert _hits_are_rrf_mode(hits) is True

    def test_mixed_modes(self) -> None:
        hits = [_make_rrf_hit(1, 1), _make_hit(2, 0.52)]
        assert _hits_are_rrf_mode(hits) is False

    def test_all_dense(self) -> None:
        hits = [_make_hit(1, 0.52), _make_hit(2, 0.48)]
        assert _hits_are_rrf_mode(hits) is False


# ---------------------------------------------------------------------------
# AC-3: score-floor gate — all hits below floor → deflection
# ---------------------------------------------------------------------------

class TestScoreFloorGate:
    def _run_with_hits(self, hits: list[dict], llm_recs: list[dict]) -> dict:
        """Run synthesize_node with mocked LLM returning llm_recs."""
        import json
        state = _make_state(hits)
        settings = _settings()

        fake_response = MagicMock()
        fake_response.usage_metadata = {"input_tokens": 10, "output_tokens": 20}
        fake_response.response_metadata = {}

        with (
            patch("agent.nodes.synthesize.ChatOpenAI") as MockLLM,
            patch("agent.nodes.synthesize.JsonOutputParser") as MockParser,
        ):
            mock_llm_instance = MagicMock()
            mock_llm_instance.invoke.return_value = fake_response
            MockLLM.return_value = mock_llm_instance

            mock_parser_instance = MagicMock()
            mock_parser_instance.invoke.return_value = llm_recs
            MockParser.return_value = mock_parser_instance

            return synthesize_node(state, settings)

    def test_all_below_floor_yields_deflection(self) -> None:
        """All hits below SCORE_FLOOR → empty recs + deflection answer."""
        hits = [
            _make_hit(101, SCORE_FLOOR - 0.05),
            _make_hit(102, SCORE_FLOOR - 0.10),
        ]
        llm_recs = [
            {"tmdb_id": 101, "title": "Film 101", "year": 2020, "why_for_you": "Great!", "provider_hint": None},
        ]
        result = self._run_with_hits(hits, llm_recs)

        assert result["recs"] == [], f"Expected empty recs, got {result['recs']}"
        assert result["final_answer"] == _DEFLECTION_ANSWER

    def test_all_below_floor_empty_recs(self) -> None:
        """Confirm recs list is [] not None when gate fires."""
        hits = [_make_hit(1, 0.10), _make_hit(2, 0.15)]
        result = self._run_with_hits(hits, [])
        assert result["recs"] == []

    def test_above_floor_hits_unchanged_behaviour(self) -> None:
        """Hits above SCORE_FLOOR → normal recommendation flow."""
        hits = [
            _make_hit(201, SCORE_FLOOR + 0.10),
            _make_hit(202, SCORE_FLOOR + 0.05),
        ]
        llm_recs = [
            {"tmdb_id": 201, "title": "Film 201", "year": 2021, "why_for_you": "Good match.", "provider_hint": None},
        ]
        result = self._run_with_hits(hits, llm_recs)

        assert result["final_answer"] != _DEFLECTION_ANSWER
        assert len(result["recs"]) == 1
        assert result["recs"][0]["tmdb_id"] == 201

    def test_mixed_hits_only_above_floor_survive(self) -> None:
        """Mixed hits: only above-floor hits pass through to recs."""
        hits = [
            _make_hit(301, SCORE_FLOOR + 0.08),   # above
            _make_hit(302, SCORE_FLOOR - 0.05),   # below
        ]
        llm_recs = [
            {"tmdb_id": 301, "title": "Film 301", "year": 2022, "why_for_you": "Great.", "provider_hint": None},
            {"tmdb_id": 302, "title": "Film 302", "year": 2023, "why_for_you": "OK.", "provider_hint": None},
        ]
        result = self._run_with_hits(hits, llm_recs)

        rec_ids = [r["tmdb_id"] for r in result["recs"]]
        assert 301 in rec_ids
        assert 302 not in rec_ids

    def test_rrf_mode_skips_floor(self) -> None:
        """RRF-mode hits are not filtered by SCORE_FLOOR (non-semantic scores)."""
        hits = [_make_rrf_hit(401, 1), _make_rrf_hit(402, 2)]
        llm_recs = [
            {"tmdb_id": 401, "title": "Film 401", "year": 2022, "why_for_you": "Fine.", "provider_hint": None},
        ]
        result = self._run_with_hits(hits, llm_recs)

        # Should not deflect — RRF mode passes through
        assert result["final_answer"] != _DEFLECTION_ANSWER
        assert len(result["recs"]) == 1


# ---------------------------------------------------------------------------
# AC-4: end-to-end "Sinval troncho" mocked test
# ---------------------------------------------------------------------------

class TestEndToEndGibberish:
    """Mocked E2E: gibberish query → below-floor hits → zero recs."""

    def test_sinval_troncho_yields_zero_recs(self) -> None:
        """Gibberish query hits (all below floor) → deflection, zero recs."""
        # Simulate what Qdrant returns for "Sinval troncho" in dense mode:
        # top-1 score was 0.324 in calibration — well below SCORE_FLOOR=0.40
        sinval_hits = [
            _make_hit(99001, 0.324),
            _make_hit(99002, 0.314),
            _make_hit(99003, 0.312),
        ]
        state = _make_state(sinval_hits, user_query="Sinval troncho")

        fake_response = MagicMock()
        fake_response.usage_metadata = {"input_tokens": 10, "output_tokens": 30}
        fake_response.response_metadata = {}

        llm_recs = [
            {"tmdb_id": 99001, "title": "Some Film", "year": 2019, "why_for_you": "Matched.", "provider_hint": None},
        ]

        with (
            patch("agent.nodes.synthesize.ChatOpenAI") as MockLLM,
            patch("agent.nodes.synthesize.JsonOutputParser") as MockParser,
        ):
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = fake_response
            MockLLM.return_value = mock_llm

            mock_parser = MagicMock()
            mock_parser.invoke.return_value = llm_recs
            MockParser.return_value = mock_parser

            result = synthesize_node(state, _settings())

        assert result["recs"] == []
        assert result["final_answer"] == _DEFLECTION_ANSWER

    def test_golden_query_yields_normal_recs(self) -> None:
        """A golden-set style query (above-floor hits) → normal recs, not deflection."""
        # Golden dense scores ranged 0.44–0.60; simulate high-confidence hits
        golden_hits = [
            _make_hit(550, 0.55),   # Fight Club (golden set)
            _make_hit(278, 0.52),   # Shawshank
            _make_hit(238, 0.50),   # Godfather
        ]
        state = _make_state(
            golden_hits,
            user_query="I'm looking for a dark psychological thriller",
        )

        fake_response = MagicMock()
        fake_response.usage_metadata = {"input_tokens": 20, "output_tokens": 50}
        fake_response.response_metadata = {}

        llm_recs = [
            {"tmdb_id": 550, "title": "Fight Club", "year": 1999, "why_for_you": "Matches.", "provider_hint": None},
            {"tmdb_id": 278, "title": "The Shawshank Redemption", "year": 1994, "why_for_you": "Fits.", "provider_hint": None},
        ]

        with (
            patch("agent.nodes.synthesize.ChatOpenAI") as MockLLM,
            patch("agent.nodes.synthesize.JsonOutputParser") as MockParser,
        ):
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = fake_response
            MockLLM.return_value = mock_llm

            mock_parser = MagicMock()
            mock_parser.invoke.return_value = llm_recs
            MockParser.return_value = mock_parser

            result = synthesize_node(state, _settings())

        assert result["final_answer"] != _DEFLECTION_ANSWER
        assert len(result["recs"]) == 2
        rec_ids = [r["tmdb_id"] for r in result["recs"]]
        assert 550 in rec_ids
        assert 278 in rec_ids


# ---------------------------------------------------------------------------
# AC-5: prompt delimiting — synthesize prompts wrap user_query and rag_hits
# ---------------------------------------------------------------------------

class TestPromptDelimiters:
    def _load_prompt_template(self, name: str) -> str:
        from agent.nodes import load_prompt
        return load_prompt(name)

    @pytest.mark.parametrize("prompt_name", ["synthesize_v2", "synthesize"])
    def test_synthesize_prompt_wraps_user_query(self, prompt_name: str) -> None:
        """The synthesize prompt template wraps {user_query} in XML delimiters."""
        template = self._load_prompt_template(prompt_name)
        # Must contain an opening tag for user_query
        assert "<user_query>" in template, (
            f"{prompt_name}: missing <user_query> delimiter"
        )
        assert "</user_query>" in template, (
            f"{prompt_name}: missing </user_query> delimiter"
        )

    @pytest.mark.parametrize("prompt_name", ["synthesize_v2", "synthesize"])
    def test_synthesize_prompt_wraps_rag_hits(self, prompt_name: str) -> None:
        """The synthesize prompt template wraps {rag_hits} in XML delimiters."""
        template = self._load_prompt_template(prompt_name)
        assert "<rag_hits>" in template, (
            f"{prompt_name}: missing <rag_hits> delimiter"
        )
        assert "</rag_hits>" in template, (
            f"{prompt_name}: missing </rag_hits> delimiter"
        )

    @pytest.mark.parametrize("prompt_name", ["synthesize_v2", "synthesize"])
    def test_synthesize_prompt_format_keys_unchanged(self, prompt_name: str) -> None:
        """format() keys {user_query} and {rag_hits} are still present after delimiting."""
        template = self._load_prompt_template(prompt_name)
        assert "{user_query}" in template, f"{prompt_name}: missing {{user_query}} format key"
        assert "{rag_hits}" in template, f"{prompt_name}: missing {{rag_hits}} format key"

    def test_rendered_synthesize_v2_contains_delimiter_markers(self) -> None:
        """Rendered prompt (after .format()) still contains the tag markers."""
        from agent.nodes import load_prompt
        template = load_prompt("synthesize_v2")
        rendered = template.format(
            rag_hits='[{"tmdb_id": 1}]',
            web_hits="[]",
            user_query="dark thriller",
            taste_top_films="none",
        )
        assert "<user_query>" in rendered
        assert "</user_query>" in rendered
        assert "<rag_hits>" in rendered
        assert "</rag_hits>" in rendered
