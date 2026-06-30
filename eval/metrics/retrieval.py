"""Pure retrieval metric functions: precision@k, recall@k, MRR, nDCG@k."""
from __future__ import annotations

import math


def precision_at_k(retrieved_ids: list[int], relevant_ids: set[int], k: int) -> float:
    """Fraction of top-k retrieved that are relevant."""
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for i in top_k if i in relevant_ids)
    return hits / k


def recall_at_k(retrieved_ids: list[int], relevant_ids: set[int], k: int) -> float:
    """Fraction of relevant items found in top-k."""
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for i in top_k if i in relevant_ids)
    return hits / len(relevant_ids)


def mrr(retrieved_ids: list[int], relevant_ids: set[int], k: int) -> float:
    """Mean Reciprocal Rank — rank of first relevant hit in top-k."""
    for rank, item_id in enumerate(retrieved_ids[:k], start=1):
        if item_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: list[int], relevant_ids: set[int], k: int) -> float:
    """Normalized Discounted Cumulative Gain at k."""

    def dcg(ids: list[int]) -> float:
        return sum(
            1.0 / math.log2(i + 2)
            for i, item_id in enumerate(ids)
            if item_id in relevant_ids
        )

    top_k = retrieved_ids[:k]
    actual_dcg = dcg(top_k)
    # Ideal: all relevant items at top positions
    ideal_count = min(len(relevant_ids), k)
    ideal_dcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_count))
    if ideal_dcg == 0.0:
        return 0.0
    return actual_dcg / ideal_dcg
