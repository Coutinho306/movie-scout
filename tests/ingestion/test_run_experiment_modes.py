"""Unit tests for run_experiment ingest-mode routing.

Verifies:
- corpus mode (--corpus): sample=False, targets unsuffixed default collections
- golden/sample path (--tmdb-ids without --corpus): sample=True,
  targets calib_-prefixed collections

Strategy: patch sys.argv, then inspect the settings object that main()
builds by intercepting run_pipeline at the module attribute level (after
any reload so the bound name is what main() calls).
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch


def _call_main_with(argv: list[str], extra_context: dict | None = None) -> dict:
    """Reload run_experiment with *argv* and capture the run_pipeline call kwargs.

    *extra_context* is a dict of ``{target: replacement}`` for additional patches
    applied before main() runs.
    """
    import ingestion.scripts.run_experiment as re_mod

    # Reload so argparse sees fresh argv.
    importlib.reload(re_mod)

    captured: list[dict] = []

    def fake_run_pipeline(**kwargs):  # type: ignore[no-untyped-def]
        captured.append(kwargs)

    # Patch the name in the already-reloaded module's namespace.
    re_mod.run_pipeline = fake_run_pipeline  # type: ignore[assignment]

    if extra_context:
        for attr_path, value in extra_context.items():
            # Support simple dotted module paths like "ingestion.scripts.build_corpus_sample.CORPUS_CACHE"
            parts = attr_path.rsplit(".", 1)
            if len(parts) == 2:
                import importlib as il
                mod = il.import_module(parts[0])
                setattr(mod, parts[1], value)

    with (
        patch("sys.argv", ["run_experiment"] + argv),
        patch.dict(
            "os.environ",
            {
                "TMDB_API_KEY": "fake_tmdb",
                "OPENAI_API_KEY": "fake_openai",
                "QDRANT_URL": "http://localhost:6333",
                "QDRANT_API_KEY": "",
            },
        ),
    ):
        re_mod.main()

    assert len(captured) == 1, f"run_pipeline should be called once; got {len(captured)}"
    return captured[0]


# ---------------------------------------------------------------------------
# Corpus mode
# ---------------------------------------------------------------------------


def test_corpus_mode_targets_default_collections() -> None:
    """--corpus with --tmdb-ids must use sample=False → unsuffixed collections."""
    call = _call_main_with(["--corpus", "--tmdb-ids", "550,680"])
    settings = call["settings"]

    assert settings.sample is False, "corpus mode must not set sample=True"
    assert settings.movies_collection == "tmdb_movies", (
        f"corpus mode must target unsuffixed collection, got {settings.movies_collection!r}"
    )
    assert settings.reviews_collection == "tmdb_reviews", (
        f"corpus mode must target unsuffixed collection, got {settings.reviews_collection!r}"
    )
    assert call.get("explicit_tmdb_ids") == [550, 680]


def test_corpus_mode_loads_corpus_json_when_no_ids(tmp_path: Path) -> None:
    """--corpus without --tmdb-ids reads from data/corpus_sample.json via load_corpus."""
    corpus_ids = [101, 202, 303]
    cache_file = tmp_path / "corpus_sample.json"
    cache_file.write_text(json.dumps({"tmdb_ids": corpus_ids, "tier_counts": {}}))

    import ingestion.scripts.build_corpus_sample as bcs_mod
    original_cache = bcs_mod.CORPUS_CACHE
    bcs_mod.CORPUS_CACHE = cache_file
    try:
        call = _call_main_with(["--corpus"])
    finally:
        bcs_mod.CORPUS_CACHE = original_cache

    settings = call["settings"]
    assert settings.sample is False
    assert settings.movies_collection == "tmdb_movies"
    assert call.get("explicit_tmdb_ids") == corpus_ids


# ---------------------------------------------------------------------------
# Golden / sample path still forces calib_ namespace
# ---------------------------------------------------------------------------


def test_tmdb_ids_without_corpus_forces_calib() -> None:
    """--tmdb-ids without --corpus must set sample=True → calib_-prefixed collections."""
    call = _call_main_with(["--tmdb-ids", "550,680"])
    settings = call["settings"]

    assert settings.sample is True, "non-corpus explicit ids must force sample=True"
    assert "calib_" in settings.movies_collection, (
        f"expected calib_ prefix, got {settings.movies_collection!r}"
    )
    assert call.get("explicit_tmdb_ids") == [550, 680]
