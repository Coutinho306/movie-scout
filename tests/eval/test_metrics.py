"""Unit tests for retrieval metric functions — no IO."""
import pytest

from eval.metrics.retrieval import mrr, ndcg_at_k, precision_at_k, recall_at_k

RETRIEVED = [10, 20, 30, 40, 50]
RELEVANT = {10, 40}


def test_precision_at_k_hit_first():
    assert precision_at_k([10, 99], {10}, 2) == 0.5


def test_recall_at_k_partial():
    assert recall_at_k(RETRIEVED, RELEVANT, 3) == 0.5  # only 10 in top-3


def test_mrr_first_hit_rank_2():
    assert mrr([99, 10, 20], {10}, 5) == pytest.approx(0.5)


def test_ndcg_at_k_perfect():
    score = ndcg_at_k([10], {10}, 3)
    assert score == pytest.approx(1.0)


def test_precision_at_k_no_hits():
    assert precision_at_k([1, 2, 3], {99}, 3) == 0.0


def test_recall_at_k_empty_retrieved():
    assert recall_at_k([], {10}, 5) == 0.0
