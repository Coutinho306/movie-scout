"""Tests for retrieval.rerank.cross_encode_rerank."""

from retrieval.models import ReviewHit
from retrieval.rerank import cross_encode_rerank


def _make_review(tmdb_id: int, text: str, score: float) -> ReviewHit:
    return ReviewHit(
        tmdb_id=tmdb_id,
        title=f"Film {tmdb_id}",
        review_author="critic",
        chunk_text=text,
        chunk_index=0,
        score=score,
    )


def test_rerank_returns_same_hits() -> None:
    hits = [
        _make_review(1, "A haunting slow film about time and memory", 0.7),
        _make_review(2, "Fast paced action blockbuster with explosions", 0.8),
        _make_review(3, "Quiet contemplative study of rural life", 0.6),
    ]
    query = "slow meditative quiet contemplative"
    ranked = cross_encode_rerank(query, hits)
    assert len(ranked) == 3
    # The quiet/contemplative/slow reviews should rank above the action one
    ids = [h.tmdb_id for h in ranked]  # type: ignore[attr-defined]
    action_pos = ids.index(2)
    assert action_pos > 0  # action film not first


def test_rerank_empty() -> None:
    assert cross_encode_rerank("query", []) == []
