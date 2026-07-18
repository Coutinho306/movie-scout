"""AC3 — --skip-reviews loads movies only, never touches the reviews collection.

On the explicit_tmdb_ids (calibration sample) path with the loaders and Qdrant /
embedder helpers mocked at the pipeline boundary:

- skip_reviews=True  → load_tmdb_reviews is NEVER called; load_tmdb_movies IS.
- skip_reviews=False → load_tmdb_reviews IS called (today's behavior preserved).

Because the review loader is never entered under skip_reviews, the live
tmdb_reviews collection receives zero writes on a skip-reviews run.

All Qdrant / TMDB / OpenAI clients are mocked — no real network, no real collections.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ingestion.config import Settings
from ingestion.pipeline import run_pipeline


def _run(*, skip_reviews: bool, monkeypatch: pytest.MonkeyPatch) -> tuple[MagicMock, MagicMock]:
    """Run the sample path with all IO mocked; return (movies_loader, reviews_loader) spies."""
    settings = Settings()
    with (
        patch("ingestion.pipeline.get_qdrant_client", return_value=MagicMock()),
        patch("ingestion.pipeline.get_embedder", return_value=MagicMock()),
        patch("ingestion.pipeline.ensure_collections"),
        patch("ingestion.pipeline.rebuild_collections"),
        patch("ingestion.pipeline.load_tmdb_movies", return_value=3) as movies,
        patch("ingestion.pipeline.load_tmdb_reviews", return_value=9) as reviews,
    ):
        # run_pipeline writes openai_api_key into os.environ["OPENAI_API_KEY"] as a
        # real side effect (get_embedder reads it from env) — monkeypatch.setenv
        # ensures pytest restores the real key after this test, so later tests
        # that need a real OpenAI call don't inherit this test's fake "x" key.
        monkeypatch.setenv("OPENAI_API_KEY", "x")
        run_pipeline(
            tmdb_api_key="x",
            openai_api_key="x",
            qdrant_url="http://localhost:6333",
            qdrant_api_key="",
            settings=settings,
            explicit_tmdb_ids=[1, 2, 3],
            skip_reviews=skip_reviews,
        )
    return movies, reviews


def test_skip_reviews_true_does_not_call_review_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    movies, reviews = _run(skip_reviews=True, monkeypatch=monkeypatch)
    reviews.assert_not_called()
    movies.assert_called_once()


def test_skip_reviews_default_off_calls_review_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    movies, reviews = _run(skip_reviews=False, monkeypatch=monkeypatch)
    reviews.assert_called_once()
    movies.assert_called_once()
