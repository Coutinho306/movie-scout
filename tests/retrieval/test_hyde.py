"""Tests for retrieval.hyde — HyDE query expansion."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from retrieval.hyde import generate_hyde_text, hyde_embed


# ---------------------------------------------------------------------------
# generate_hyde_text
# ---------------------------------------------------------------------------


def test_generate_hyde_text_returns_string() -> None:
    """Live integration test: calls real OpenAI API, should return non-empty str."""
    result = generate_hyde_text("a slow burn psychological thriller about grief")
    assert isinstance(result, str)
    assert len(result) > 0


def test_generate_hyde_text_cached() -> None:
    """Second call with same query must return the cached string (same object)."""
    q = "sci-fi film about time travel and regret"
    r1 = generate_hyde_text(q)
    r2 = generate_hyde_text(q)
    assert r1 is r2


def test_generate_hyde_text_fallback_on_error() -> None:
    """On API error, generate_hyde_text must fall back to the raw query."""
    # Use a unique query to avoid a warm cache hit from other tests.
    unique_q = "fallback_test_query_xyz_unique_string_99887"

    with patch("retrieval.hyde._get_client") as mock_client_factory:
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("api down")
        mock_client_factory.return_value = mock_client

        result = generate_hyde_text.__wrapped__(unique_q)  # bypass lru_cache

    assert result == unique_q


# ---------------------------------------------------------------------------
# hyde_embed
# ---------------------------------------------------------------------------


def test_hyde_embed_pure_returns_list() -> None:
    """Pure HyDE (no blend) must return a list of floats."""
    mock_embedder = MagicMock()
    mock_embedder.embed_single.return_value = [0.1, 0.2, 0.3]

    with patch("retrieval.hyde.generate_hyde_text", return_value="A man discovers"):
        result = hyde_embed("moody noir thriller", mock_embedder, blend_alpha=None)

    assert isinstance(result, list)
    assert len(result) == 3
    # embed_single called once (for the hypothetical text)
    mock_embedder.embed_single.assert_called_once_with("A man discovers")


def test_hyde_embed_blend_averages_vectors() -> None:
    """With blend_alpha=0.5, result must be the midpoint of query and hyde vecs."""
    mock_embedder = MagicMock()
    # First call: hyde text, second call: raw query
    mock_embedder.embed_single.side_effect = [
        [0.0, 1.0],   # hyde vector
        [1.0, 0.0],   # query vector
    ]

    with patch("retrieval.hyde.generate_hyde_text", return_value="Hypothetical text"):
        result = hyde_embed("my query", mock_embedder, blend_alpha=0.5)

    assert result == pytest.approx([0.5, 0.5])
    assert mock_embedder.embed_single.call_count == 2
