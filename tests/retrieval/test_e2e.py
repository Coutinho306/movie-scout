"""End-to-end smoke test: all retrieval primitives, hybrid+rerank+rewrite enabled."""

import pytest

from retrieval.config import RetrievalSettings
from retrieval.movies import search_movies
from retrieval.rerank import cross_encode_rerank
from retrieval.rewrite import rewrite_query
from retrieval.reviews import search_reviews
from retrieval.taste import score_against_taste


def test_full_retrieval_pipeline() -> None:
    settings = RetrievalSettings(hybrid=True, rerank=True, query_rewrite=True, top_k=10)

    raw_query = "recommend something slow and meditative"

    # 1. query rewrite
    query = rewrite_query(raw_query)
    assert isinstance(query, str) and len(query) > 0

    # 2. movies search
    movie_hits = search_movies(query, settings=settings, k=10)
    assert isinstance(movie_hits, list)

    # 3. reviews search
    review_hits = search_reviews(query, settings=settings, k=10)
    assert isinstance(review_hits, list)

    # 4. taste re-scoring (no explicit profile → loads from disk)
    if movie_hits:
        taste_hits = score_against_taste(movie_hits)
        assert len(taste_hits) == len(movie_hits)
        scores = [h.blended_score for h in taste_hits]
        assert scores == sorted(scores, reverse=True)

    # 5. reranking
    if review_hits:
        reranked = cross_encode_rerank(query, review_hits)
        assert len(reranked) == len(review_hits)
