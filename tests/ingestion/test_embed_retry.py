"""Unit tests for OpenAIEmbedder.embed_texts retry-on-transient-error logic.

429 (rate limit) and 5xx (transient server errors) must be retried with
bounded exponential backoff; 4xx client errors must propagate immediately
without consuming retry budget; exhaustion must re-raise the last error.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest
from openai import APIStatusError

from ingestion.embedding import OpenAIEmbedder, _MAX_ATTEMPTS


def _make_status_error(status_code: int) -> APIStatusError:
    request = httpx.Request("POST", "https://api.openai.com/v1/embeddings")
    response = httpx.Response(status_code, request=request)
    return APIStatusError("simulated", response=response, body=None)


def _make_success_response(n: int = 2) -> MagicMock:
    """Return a mock openai embeddings response with n dummy embeddings."""
    response = MagicMock()
    response.usage.total_tokens = n * 10
    response.data = [MagicMock(embedding=[0.1, 0.2]) for _ in range(n)]
    return response


def _embedder() -> OpenAIEmbedder:
    return OpenAIEmbedder(model="text-embedding-3-small", dim=1536)


# ---------------------------------------------------------------------------
# Retry-then-succeed: 429 / 5xx must retry and eventually return data
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("status_code", [429, 500, 503])
def test_retryable_error_then_succeeds(status_code: int) -> None:
    """A single retryable error must be retried; success on second attempt returns data."""
    call_count = 0

    def _fake_create(**kwargs: object) -> MagicMock:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise _make_status_error(status_code)
        return _make_success_response()

    embedder = _embedder()
    mock_client = MagicMock()
    mock_client.embeddings.create.side_effect = _fake_create

    with (
        patch.object(embedder, "_get_client", return_value=mock_client),
        patch("ingestion.embedding.time.sleep"),
    ):
        result = embedder.embed_texts(["hello", "world"])

    assert call_count == 2, f"Expected 2 calls (1 fail + 1 success), got {call_count}"
    assert len(result) == 2
    assert result[0] == [0.1, 0.2]


# ---------------------------------------------------------------------------
# Non-retryable 4xx: must propagate immediately with no retry
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("status_code", [400, 401, 403])
def test_non_retryable_4xx_propagates_immediately(status_code: int) -> None:
    """A non-retryable 4xx error must raise on the first attempt, no retry."""
    call_count = 0

    def _fake_create(**kwargs: object) -> MagicMock:
        nonlocal call_count
        call_count += 1
        raise _make_status_error(status_code)

    embedder = _embedder()
    mock_client = MagicMock()
    mock_client.embeddings.create.side_effect = _fake_create

    with (
        patch.object(embedder, "_get_client", return_value=mock_client),
        patch("ingestion.embedding.time.sleep") as mock_sleep,
    ):
        with pytest.raises(APIStatusError) as exc_info:
            embedder.embed_texts(["hello"])

    assert call_count == 1, f"4xx must not retry; got {call_count} calls"
    mock_sleep.assert_not_called()
    assert exc_info.value.status_code == status_code


# ---------------------------------------------------------------------------
# Exhaustion: all retries fail — must re-raise last error, not swallow it
# ---------------------------------------------------------------------------

def test_exhaustion_reraises_last_error() -> None:
    """When all _MAX_ATTEMPTS are consumed, the last APIStatusError is re-raised."""
    call_count = 0

    def _fake_create(**kwargs: object) -> MagicMock:
        nonlocal call_count
        call_count += 1
        raise _make_status_error(429)

    embedder = _embedder()
    mock_client = MagicMock()
    mock_client.embeddings.create.side_effect = _fake_create

    with (
        patch.object(embedder, "_get_client", return_value=mock_client),
        patch("ingestion.embedding.time.sleep"),
    ):
        with pytest.raises(APIStatusError) as exc_info:
            embedder.embed_texts(["hello"])

    assert call_count == _MAX_ATTEMPTS, (
        f"Expected exactly {_MAX_ATTEMPTS} attempts on exhaustion, got {call_count}"
    )
    assert exc_info.value.status_code == 429


# ---------------------------------------------------------------------------
# Backoff timing: sleep durations follow base * 2^(attempt-1) pattern
# ---------------------------------------------------------------------------

def test_429_retry_uses_exponential_backoff() -> None:
    """Verify sleep durations follow 2.0 * 2^(attempt-1) for retryable errors."""
    call_count = 0

    def _fake_create(**kwargs: object) -> MagicMock:
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise _make_status_error(429)
        return _make_success_response()

    embedder = _embedder()
    mock_client = MagicMock()
    mock_client.embeddings.create.side_effect = _fake_create
    sleep_calls: list[float] = []

    with (
        patch.object(embedder, "_get_client", return_value=mock_client),
        patch("ingestion.embedding.time.sleep", side_effect=lambda s: sleep_calls.append(s)),
    ):
        embedder.embed_texts(["hello"])

    # First failure → sleep(2.0 * 2^0 = 2.0), second → sleep(2.0 * 2^1 = 4.0)
    assert sleep_calls == [2.0, 4.0], (
        f"Expected exponential backoff [2.0, 4.0], got {sleep_calls}"
    )
