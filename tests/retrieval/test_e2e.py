"""End-to-end smoke test: all retrieval primitives with rerank+rewrite enabled.

Note on hybrid: search_movies supports hybrid (restored in specs/0009);
search_reviews is dense-only (tmdb_reviews has no sparse field). This test
uses hybrid=True for movies to exercise the restored RRF path, and
separately uses default settings for reviews (dense-only).
"""

from pathlib import Path

import pytest

from ingestion.scripts.compute_taste import load_taste_profile
from retrieval.config import RetrievalSettings
from retrieval.movies import search_movies
from retrieval.rerank import cross_encode_rerank
from retrieval.rewrite import rewrite_query
from retrieval.reviews import search_reviews
from retrieval.taste import score_against_taste


def test_full_retrieval_pipeline() -> None:
    # Movies: use hybrid to exercise the restored RRF path.
    movie_settings = RetrievalSettings(hybrid=True, query_rewrite=True, top_k=10)
    # Reviews: dense-only (tmdb_reviews has no sparse field).
    review_settings = RetrievalSettings(query_rewrite=True, top_k=10)

    raw_query = "recommend something slow and meditative"

    # 1. query rewrite
    query = rewrite_query(raw_query)
    assert isinstance(query, str) and len(query) > 0

    # 2. movies search (hybrid RRF)
    movie_hits = search_movies(query, settings=movie_settings, k=10)
    assert isinstance(movie_hits, list)

    # 3. reviews search (dense only)
    review_hits = search_reviews(query, settings=review_settings, k=10)
    assert isinstance(review_hits, list)

    # 4. taste re-scoring with explicit profile (offline dev profile from disk)
    if movie_hits:
        profile = load_taste_profile(Path("data/taste_profile.json"))
        taste_hits = score_against_taste(movie_hits, profile=profile)
        assert len(taste_hits) == len(movie_hits)
        scores = [h.blended_score for h in taste_hits]
        assert scores == sorted(scores, reverse=True)

    # 5. reranking
    if review_hits:
        reranked = cross_encode_rerank(query, review_hits)
        assert len(reranked) == len(review_hits)
