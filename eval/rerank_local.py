"""Eval-only cross-encoder rerank helper + retrieval re-test runner.

The ``cross_encode_rerank`` function is recovered verbatim from
``git show d55b46d^:retrieval/rerank.py`` (53 lines) and lives ONLY here.
It does NOT restore ``retrieval/rerank.py``, ``RetrievalSettings.rerank``, or
the ``retrieval/movies.py`` fetch-limit change. Spec 0020's production removal
stays intact.

Usage::

    uv run python3 -m eval.rerank_local
"""

from __future__ import annotations

import functools
import itertools
import json
import logging
import statistics
import time
from pathlib import Path
from typing import Union

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# cross_encode_rerank — recovered verbatim from d55b46d^:retrieval/rerank.py
# ---------------------------------------------------------------------------

from retrieval.models import MovieHit, ReviewHit  # noqa: E402

Hit = Union[MovieHit, ReviewHit]

_DEFAULT_MODEL = "cross-encoder/stsb-distilroberta-base"


@functools.lru_cache(maxsize=4)
def _load_cross_encoder(model: str):  # type: ignore[return]
    from sentence_transformers import CrossEncoder  # lazy — only when rerank=True

    return CrossEncoder(model)


def cross_encode_rerank(
    query: str,
    hits: list[Hit],
    *,
    model: str = _DEFAULT_MODEL,
) -> list[Hit]:
    """Reorder hits by cross-encoder score (higher = more relevant).

    Accepts either MovieHit or ReviewHit. Works on mixed lists only if
    the caller guarantees hit.chunk_text / hit.overview as the passage text.
    """
    if not hits:
        return hits

    encoder = _load_cross_encoder(model)

    def _text(hit: Hit) -> str:
        if isinstance(hit, ReviewHit):
            return hit.chunk_text
        return f"{hit.title} {hit.overview}"

    pairs = [(query, _text(h)) for h in hits]
    scores: list[float] = encoder.predict(pairs).tolist()

    ranked = sorted(zip(hits, scores), key=lambda x: x[1], reverse=True)
    logger.debug('{"step":"rerank","model":"%s","hits":%d}', model, len(hits))
    return [h for h, _ in ranked]


# ---------------------------------------------------------------------------
# Eval-only runner
# ---------------------------------------------------------------------------

_WIDENED_POOL = 50  # fetch k=50 candidates, then rerank to top_k


def _run_config(
    cfg: dict,
    golden_queries: list,
) -> tuple[dict, dict]:
    """Return (baseline_metrics, rerank_metrics) for one grid config."""
    from eval.metrics.retrieval import ndcg_at_k, recall_at_k
    from ingestion.config import Settings as IngestionSettings
    from retrieval.config import RetrievalSettings
    from retrieval.movies import search_movies

    top_k: int = cfg["top_k"]
    variant: str = cfg.get("variant", "default")

    ingestion = IngestionSettings.from_variant_suffix(variant)

    # Widened pool settings (50 hits for reranker input)
    settings_wide = RetrievalSettings(
        top_k=_WIDENED_POOL,
        hybrid=cfg["hybrid"],
        query_rewrite=cfg["query_rewrite"],
    )
    settings_wide = settings_wide.with_ingestion(ingestion)

    # Baseline settings (top_k hits, no rerank)
    settings_base = RetrievalSettings(
        top_k=top_k,
        hybrid=cfg["hybrid"],
        query_rewrite=cfg["query_rewrite"],
    )
    settings_base = settings_base.with_ingestion(ingestion)

    base_ndcg, base_recall = [], []
    rerank_ndcg, rerank_recall = [], []

    for gq in golden_queries:
        relevant = gq.target_tmdb_ids

        # Baseline: fetch top_k, no rerank
        try:
            base_hits = search_movies(gq.text, settings=settings_base, k=top_k)
        except Exception as exc:
            logger.warning("baseline search failed: %s", exc)
            base_hits = []

        base_ids = [h.tmdb_id for h in base_hits]
        base_ndcg.append(ndcg_at_k(base_ids, relevant, top_k))
        base_recall.append(recall_at_k(base_ids, relevant, top_k))

        # Rerank: fetch widened pool, rerank, slice to top_k
        try:
            wide_hits = search_movies(gq.text, settings=settings_wide, k=_WIDENED_POOL)
        except Exception as exc:
            logger.warning("widened search failed: %s", exc)
            wide_hits = []

        if wide_hits:
            reranked = cross_encode_rerank(gq.text, wide_hits)
        else:
            reranked = []
        rerank_ids = [h.tmdb_id for h in reranked[:top_k]]
        rerank_ndcg.append(ndcg_at_k(rerank_ids, relevant, top_k))
        rerank_recall.append(recall_at_k(rerank_ids, relevant, top_k))

    def _mean(vals: list[float]) -> float:
        return statistics.mean(vals) if vals else 0.0

    baseline = {
        "mean_ndcg": _mean(base_ndcg),
        "mean_recall": _mean(base_recall),
    }
    rerank = {
        "mean_ndcg": _mean(rerank_ndcg),
        "mean_recall": _mean(rerank_recall),
    }
    return baseline, rerank


def run_rerank_retest(
    grid_yaml: Path | None = None,
    results_path: Path | None = None,
) -> list[dict]:
    """Run the 8-config rerank re-test over the rebuilt multi-relevant golden set.

    Returns a list of result dicts, one per grid config, containing:
    config name, top_k, hybrid, query_rewrite, baseline nDCG@k, rerank nDCG@k, Δ.
    """
    import yaml

    from eval.golden import build_golden_set

    if grid_yaml is None:
        grid_yaml = Path("eval/grids/retrieval.yaml")
    if results_path is None:
        results_path = Path("eval/runs/rerank_retest_results.json")

    # Load the multi-relevant golden set
    golden = build_golden_set()
    logger.info("Golden set: %d queries", len(golden.queries))

    # Load grid
    with grid_yaml.open() as f:
        raw = yaml.safe_load(f)
    keys = list(raw.keys())
    values = [raw[k] for k in keys]
    configs = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    logger.info("Grid: %d configs", len(configs))

    results = []
    for idx, cfg in enumerate(configs):
        config_id = f"ret_{idx:03d}"
        config_name = (
            f"top_k={cfg['top_k']} hybrid={cfg['hybrid']} qrewrite={cfg['query_rewrite']}"
        )
        logger.info("Config %s: %s", config_id, config_name)

        t0 = time.time()
        baseline, rerank = _run_config(cfg, golden.queries)
        elapsed = time.time() - t0

        delta = rerank["mean_ndcg"] - baseline["mean_ndcg"]
        row = {
            "config_id": config_id,
            "top_k": cfg["top_k"],
            "hybrid": cfg["hybrid"],
            "query_rewrite": cfg["query_rewrite"],
            "variant": cfg.get("variant", "default"),
            "baseline_ndcg": round(baseline["mean_ndcg"], 4),
            "rerank_ndcg": round(rerank["mean_ndcg"], 4),
            "delta_ndcg": round(delta, 4),
            "baseline_recall": round(baseline["mean_recall"], 4),
            "rerank_recall": round(rerank["mean_recall"], 4),
            "elapsed_s": round(elapsed, 1),
        }
        results.append(row)
        logger.info(
            "  baseline nDCG=%.4f  rerank nDCG=%.4f  Δ=%.4f  (%.1fs)",
            baseline["mean_ndcg"],
            rerank["mean_ndcg"],
            delta,
            elapsed,
        )

    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2))
    logger.info("Results saved to %s", results_path)
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    results = run_rerank_retest()
    print("\n=== Rerank re-test summary (multi-relevant golden set) ===")
    print(
        f"{'Config':<40} {'Baseline':>10} {'Rerank':>10} {'Δ nDCG':>10}"
    )
    print("-" * 72)
    for r in results:
        label = f"k={r['top_k']} hybrid={r['hybrid']} qrw={r['query_rewrite']}"
        print(
            f"{label:<40} {r['baseline_ndcg']:>10.4f} {r['rerank_ndcg']:>10.4f} {r['delta_ndcg']:>+10.4f}"
        )
    net_delta = sum(r["delta_ndcg"] for r in results)
    positive = sum(1 for r in results if r["delta_ndcg"] > 0)
    print("-" * 72)
    print(f"Net Δ nDCG: {net_delta:+.4f}  |  Positive configs: {positive}/{len(results)}")
