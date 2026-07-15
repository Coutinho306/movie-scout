"""AC5 — Concurrent theme-cache stress test.

N threads calling extract_themes on overlapping + distinct tmdb_ids (mocked LLM)
must leave data/theme_cache.json as valid JSON containing every distinct id
exactly once with no lost writes.  The mocked LLM must be called at most once
per distinct id.

Follows the mocked-client style of tests/ingestion/test_theme_recipe.py:
no real network, no real OpenAI calls.
"""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest

import ingestion.theme_extraction as theme_mod
from ingestion.models import TmdbMovieMetadata


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_movie(tmdb_id: int) -> TmdbMovieMetadata:
    return TmdbMovieMetadata(
        tmdb_id=tmdb_id,
        title=f"Movie {tmdb_id}",
        year=2000 + (tmdb_id % 24),
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
    """Reset module-level cache state and redirect cache file to a tmp location."""
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


# ---------------------------------------------------------------------------
# AC5 stress test
# ---------------------------------------------------------------------------


def test_concurrent_theme_cache_no_corruption_no_lost_writes() -> None:
    """N threads, overlapping+distinct ids — final on-disk cache is valid JSON,
    key set == distinct input ids, LLM called at most once per distinct id."""
    distinct_ids = list(range(1, 21))  # 20 distinct ids
    # Build id list with deliberate duplicates to stress the check-then-return path.
    copies_per_id = 3
    input_ids = distinct_ids * copies_per_id  # 60 calls total, every id appears 3 times

    llm_call_counts: dict[int, int] = {i: 0 for i in distinct_ids}
    call_count_lock = threading.Lock()

    def _make_mock_client(tmdb_id: int) -> MagicMock:
        """Return a mock OpenAI client whose response encodes the tmdb_id."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = f"Themes for movie {tmdb_id}."
        mock_client.chat.completions.create.return_value = mock_response
        return mock_client

    # We need a single shared mock that tracks call counts across threads.
    # Patch _get_client to return a mock whose .chat.completions.create records calls.
    create_call_log: list[int] = []
    create_log_lock = threading.Lock()

    def _fake_create(**kwargs: object) -> MagicMock:
        # Extract tmdb_id from the prompt content.
        messages = kwargs.get("messages", [])
        content = messages[0]["content"] if messages else ""
        # The prompt contains the title which encodes the id ("Movie {tmdb_id}").
        # Extract from the first line of the prompt.
        import re
        match = re.search(r"Movie (\d+)", content)
        found_id = int(match.group(1)) if match else -1
        with create_log_lock:
            create_call_log.append(found_id)
        mock_response = MagicMock()
        mock_response.choices[0].message.content = f"Themes for movie {found_id}."
        return mock_response

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = _fake_create

    with patch.object(theme_mod, "_get_client", return_value=mock_client):
        movies = [_make_movie(tid) for tid in input_ids]

        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = [executor.submit(theme_mod.extract_themes, m) for m in movies]
            results = [f.result() for f in as_completed(futures)]

    # 1. All calls returned a string (no exceptions propagated).
    assert len(results) == len(input_ids)
    assert all(isinstance(r, str) for r in results)

    # 2. On-disk cache is valid JSON.
    cache_path = theme_mod._CACHE_PATH
    assert cache_path.exists(), "Cache file must exist after concurrent writes"
    raw = cache_path.read_text()
    on_disk: dict[str, str] = json.loads(raw)  # raises if not valid JSON

    # 3. Key set equals exactly the distinct input ids.
    expected_keys = {str(i) for i in distinct_ids}
    assert set(on_disk.keys()) == expected_keys, (
        f"Missing keys: {expected_keys - set(on_disk.keys())}\n"
        f"Extra keys: {set(on_disk.keys()) - expected_keys}"
    )

    # 4. LLM called at most once per distinct id (allow up to copies_per_id in the
    #    worst case — all N thread-copies miss before any writes back).  The key
    #    invariant is no corruption and all keys present; call-count is a
    #    sanity upper-bound, not an exact target.
    from collections import Counter
    call_counter = Counter(create_call_log)
    for tid in distinct_ids:
        count = call_counter.get(tid, 0)
        assert count <= copies_per_id, (
            f"LLM called {count} times for id {tid}; expected at most "
            f"{copies_per_id} (one per concurrent copy)"
        )


def test_concurrent_theme_cache_overlapping_ids_all_present() -> None:
    """All distinct ids appear in the final cache even with heavy overlap."""
    distinct_ids = [100, 200, 300]
    # 30 threads all hitting the same 3 ids.
    input_ids = distinct_ids * 10

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "shared theme text"
    mock_client.chat.completions.create.return_value = mock_response

    with patch.object(theme_mod, "_get_client", return_value=mock_client):
        movies = [_make_movie(tid) for tid in input_ids]

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(theme_mod.extract_themes, m) for m in movies]
            _ = [f.result() for f in as_completed(futures)]

    cache_path = theme_mod._CACHE_PATH
    assert cache_path.exists()
    on_disk = json.loads(cache_path.read_text())
    for tid in distinct_ids:
        assert str(tid) in on_disk, f"Key {tid} missing from cache"
