"""Retry-on-transient-error coverage for ingestion/theme_extraction.py.

429 (rate limit) and 5xx (transient server error) must both be retried with
backoff; 4xx client errors (bad request, auth) must fail fast with no retry.
"""

from __future__ import annotations

from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import httpx
import pytest
from openai import APIStatusError

import ingestion.theme_extraction as theme_mod
from ingestion.models import TmdbMovieMetadata


def _make_movie(tmdb_id: int = 1) -> TmdbMovieMetadata:
    return TmdbMovieMetadata(
        tmdb_id=tmdb_id,
        title="Test Movie",
        year=2020,
        overview="An overview.",
        tagline="",
        genres=["Drama"],
        cast=["Actor A"],
        director="Director D",
        runtime=100,
        vote_average=7.0,
        popularity=50.0,
        keywords=["keyword1"],
        embed_text="",
    )


@pytest.fixture(autouse=True)
def reset_theme_module_state(tmp_path: Path) -> Generator[None, None, None]:
    original_cache = theme_mod._cache
    original_cache_path = theme_mod._CACHE_PATH
    original_client = theme_mod._client

    theme_mod._cache = None
    theme_mod._CACHE_PATH = tmp_path / "theme_cache.json"
    theme_mod._client = None

    yield

    theme_mod._cache = original_cache
    theme_mod._CACHE_PATH = original_cache_path
    theme_mod._client = original_client


def _status_error(status_code: int) -> APIStatusError:
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(status_code, request=request)
    return APIStatusError("simulated", response=response, body=None)


def test_5xx_is_retried_and_eventually_succeeds() -> None:
    """A transient 500 must be retried, not fail on the first attempt."""
    call_count = 0

    def _fake_create(**kwargs: object) -> MagicMock:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise _status_error(500)
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "A real theme sentence."
        return mock_response

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = _fake_create

    with patch.object(theme_mod, "_get_client", return_value=mock_client), \
         patch.object(theme_mod.time, "sleep"):
        result = theme_mod.extract_themes(_make_movie())

    assert result == "A real theme sentence."
    assert call_count == 2, "500 should be retried once before succeeding"


def test_4xx_fails_fast_with_no_retry() -> None:
    """A non-retryable 4xx (e.g. bad request) must not consume retry budget."""
    call_count = 0

    def _fake_create(**kwargs: object) -> MagicMock:
        nonlocal call_count
        call_count += 1
        raise _status_error(400)

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = _fake_create

    with patch.object(theme_mod, "_get_client", return_value=mock_client), \
         patch.object(theme_mod.time, "sleep"):
        result = theme_mod.extract_themes(_make_movie())

    assert result == ""
    assert call_count == 1, "4xx must fail on the first attempt, no retry"
