"""LLM-based abstract thematic synthesis for movie embed_text (document-side).

Mirrors retrieval/hyde.py's shape: lazy OpenAI client, prompt loaded from file,
gpt-4o-mini, temperature=0.3, max_tokens=120.

Key difference from hyde.py: uses a PERSISTENT on-disk JSON cache at
data/theme_cache.json keyed by str(tmdb_id).  An in-process lru_cache alone
does not survive process death — which has happened repeatedly during ingest
sessions — so we need resumability across restarts.

The cache is loaded once per process (module-level dict), read-through on miss,
and written back immediately after every new LLM call so a mid-batch crash loses
at most one theme synthesis.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

from openai import OpenAI

from ingestion.models import TmdbMovieMetadata

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent / "prompts" / "theme_extraction.md"
_PROMPT_TEMPLATE = _PROMPT_PATH.read_text()

_DEFAULT_MODEL = "gpt-4o-mini"
_CACHE_PATH = Path("data/theme_cache.json")

_client: OpenAI | None = None

# Module-level in-process cache (populated from disk on first access).
_cache: dict[str, str] | None = None

# Guards the check-then-return (cache read) and the cache-write critical sections.
# The slow LLM call is intentionally kept OUTSIDE the lock so concurrent misses
# on distinct ids do not serialize on the network.  A double-miss (two threads
# miss before either writes) costs at most one redundant LLM call and is
# last-write-wins safe — the file never corrupts.
_lock = threading.Lock()


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def _load_cache() -> dict[str, str]:
    global _cache
    if _cache is None:
        if _CACHE_PATH.exists():
            try:
                _cache = json.loads(_CACHE_PATH.read_text())
            except Exception:
                logger.warning("theme_cache.json unreadable; starting fresh")
                _cache = {}
        else:
            _cache = {}
    return _cache


def _save_cache(cache: dict[str, str]) -> None:
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2))


def extract_themes(
    metadata: TmdbMovieMetadata,
    *,
    model: str = _DEFAULT_MODEL,
) -> str:
    """Return 1-3 abstract thematic sentences for the movie, or "" on failure.

    Results are cached on disk keyed by str(tmdb_id) so interrupted ingest
    runs resume without re-calling the LLM for already-processed films.
    """
    key = str(metadata.tmdb_id)

    # --- Critical section 1: check cache (fast path) ---
    with _lock:
        cache = _load_cache()
        if key in cache:
            logger.debug('{"step":"theme_cache_hit","tmdb_id":%s}', key)
            return cache[key]

    # LLM call is OUTSIDE the lock — concurrent misses on distinct ids do not
    # serialize.  A double-miss (two threads, same id) is last-write-wins safe.
    try:
        prompt = _PROMPT_TEMPLATE.format(
            title=metadata.title,
            year=metadata.year,
            genres=", ".join(metadata.genres),
            overview=metadata.overview,
            keywords=", ".join(metadata.keywords),
        )
        response = _get_client().chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=120,
            temperature=0.3,
        )
        text = (response.choices[0].message.content or "").strip()
    except Exception:
        logger.warning('{"step":"theme_llm_error","tmdb_id":%s}', key)
        return ""

    result = text if text else ""

    # --- Critical section 2: write cache (re-load in case another thread wrote) ---
    with _lock:
        cache = _load_cache()
        cache[key] = result
        _save_cache(cache)
    logger.debug('{"step":"theme_generated","tmdb_id":%s}', key)
    return result
