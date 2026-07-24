"""AC-7 confirming calibration spot-check for spec 0025.

Runs golden-set queries + item-4a gibberish probe in hybrid mode
(and HyDE-on hybrid mode) and collects dense_score distributions.
Confirms SCORE_FLOOR=0.40 separates them.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

GOLDEN_PATH = Path("data/golden_set.json")
SCORE_FLOOR = 0.40

GIBBERISH_QUERIES = [
    # item-4a short gibberish
    "Sinval troncho",
    "asdfghjkl",
    # hybrid-shaped long gibberish (> 8 tokens, bypassed gate pre-fix)
    "asdfghjkl film — something moody and atmospheric with rain and fog",
    "xqzwpvbn krtlmdf — a deeply existential journey through nothingness",
    "zzzzz qqqq rrrr — mysterious creatures in a surreal dreamlike landscape",
]


def _cosine_range(scores: list[float]) -> str:
    if not scores:
        return "n/a"
    return f"min={min(scores):.4f}  max={max(scores):.4f}  mean={statistics.mean(scores):.4f}  P10={statistics.quantiles(scores, n=10)[0]:.4f}"


def run_calibration(query_rewrite: bool = False) -> None:
    from retrieval.config import RetrievalSettings
    from retrieval.movies import search_movies

    label = "HyDE-ON hybrid" if query_rewrite else "hybrid"
    settings = RetrievalSettings(hybrid=True, query_rewrite=query_rewrite)

    golden_data = json.loads(GOLDEN_PATH.read_text())
    queries = [q["text"] for q in golden_data["queries"]]

    print(f"\n{'='*60}")
    print(f"Mode: {label}")
    print(f"{'='*60}")

    # Golden set
    golden_top1: list[float] = []
    golden_all: list[float] = []
    print(f"\nRunning {len(queries)} golden queries...")
    for q in queries:
        hits = search_movies(q, settings=settings, k=10)
        if hits:
            golden_top1.append(hits[0].dense_score)
            golden_all.extend(h.dense_score for h in hits)
    print(f"  Golden top-1 dense_score: {_cosine_range(golden_top1)}")
    print(f"  Golden all-hits dense_score: {_cosine_range(golden_all)}")
    all_golden_above = all(s >= SCORE_FLOOR for s in golden_top1)
    print(f"  All golden top-1 above {SCORE_FLOOR}: {all_golden_above}")

    # Gibberish
    gibberish_top1: list[float] = []
    gibberish_all: list[float] = []
    print(f"\nRunning {len(GIBBERISH_QUERIES)} gibberish queries...")
    for q in GIBBERISH_QUERIES:
        hits = search_movies(q, settings=settings, k=10)
        print(f"  Query: '{q[:60]}'")
        if hits:
            top1 = hits[0].dense_score
            gibberish_top1.append(top1)
            gibberish_all.extend(h.dense_score for h in hits)
            print(f"    top-1 dense_score={top1:.4f}  (below floor: {top1 < SCORE_FLOOR})")
        else:
            print("    no hits")
    print(f"\n  Gibberish top-1 dense_score: {_cosine_range(gibberish_top1)}")
    print(f"  All gibberish top-1 below {SCORE_FLOOR}: {all(s < SCORE_FLOOR for s in gibberish_top1)}")

    # Separation
    if golden_top1 and gibberish_top1:
        margin = min(golden_top1) - max(gibberish_top1)
        print(f"\n  Separation margin (min_golden - max_gibberish): {margin:.4f}")
        if margin > 0:
            print(f"  => CLEAR SEPARATION — SCORE_FLOOR={SCORE_FLOOR} transfers to {label} mode.")
        else:
            print(f"  => UNDER-SEPARATION — follow-up needed (see STATUS.md).")

    return {
        "mode": label,
        "golden_top1": golden_top1,
        "gibberish_top1": gibberish_top1,
        "all_golden_above_floor": all_golden_above,
        "all_gibberish_below_floor": all(s < SCORE_FLOOR for s in gibberish_top1),
        "margin": min(golden_top1) - max(gibberish_top1) if golden_top1 and gibberish_top1 else None,
    }


if __name__ == "__main__":
    results_dense = run_calibration(query_rewrite=False)
    results_hyde = run_calibration(query_rewrite=True)
