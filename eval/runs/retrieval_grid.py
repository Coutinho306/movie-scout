"""Grid search over retrieval knobs; writes CSV + best_retrieval.json."""
from __future__ import annotations

import csv
import itertools
import json
import logging
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from eval.golden import GoldenSet, build_golden_set
from eval.metrics.retrieval import mrr, ndcg_at_k, precision_at_k, recall_at_k
from retrieval.config import RetrievalSettings
from retrieval.movies import search_movies

logger = logging.getLogger(__name__)
RUNS_DIR = Path("eval/runs")
DEFAULT_GRID = Path("eval/grids/retrieval.yaml")
BEST_PATH = Path("eval/runs/best_retrieval.json")


def _load_grid(grid_yaml: Path) -> list[dict]:
    with grid_yaml.open() as f:
        raw = yaml.safe_load(f)
    keys = list(raw.keys())
    values = [raw[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def _run_config(cfg: dict, golden: GoldenSet) -> dict:
    variant = cfg.get("variant", "default")
    _collection_override = None if variant == "default" else f"tmdb_movies__{variant}"

    settings = RetrievalSettings(
        top_k=cfg["top_k"],
        hybrid=cfg["hybrid"],
        rerank=cfg["rerank"],
        query_rewrite=cfg["query_rewrite"],
    )

    latencies: list[float] = []
    p_vals, r_vals, mrr_vals, ndcg_vals = [], [], [], []

    for gq in golden.queries:
        t0 = time.time()
        try:
            hits = search_movies(gq.text, settings=settings, k=cfg["top_k"])
        except Exception as exc:  # noqa: BLE001 — record empty result, keep grid running
            logger.warning("search_movies failed for query %r: %s", gq.text, exc)
            hits = []
        latency_ms = (time.time() - t0) * 1000
        latencies.append(latency_ms)

        retrieved_ids = [h.tmdb_id for h in hits]
        relevant = gq.target_tmdb_ids

        p_vals.append(precision_at_k(retrieved_ids, relevant, cfg["top_k"]))
        r_vals.append(recall_at_k(retrieved_ids, relevant, cfg["top_k"]))
        mrr_vals.append(mrr(retrieved_ids, relevant, cfg["top_k"]))
        ndcg_vals.append(ndcg_at_k(retrieved_ids, relevant, cfg["top_k"]))

    latencies_sorted = sorted(latencies)
    p50_idx = len(latencies_sorted) // 2
    latency_p50 = latencies_sorted[p50_idx] if latencies_sorted else 0.0

    return {
        "mean_precision_at_k": statistics.mean(p_vals) if p_vals else 0.0,
        "mean_recall_at_k": statistics.mean(r_vals) if r_vals else 0.0,
        "mean_mrr": statistics.mean(mrr_vals) if mrr_vals else 0.0,
        "mean_ndcg_at_k": statistics.mean(ndcg_vals) if ndcg_vals else 0.0,
        "latency_p50_ms": round(latency_p50, 1),
        "cost_usd": 0.0,  # retrieval is free (Qdrant local/cloud)
    }


def run(grid_yaml: Path = DEFAULT_GRID) -> Path:
    """Run retrieval grid; return path to output CSV."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    golden = build_golden_set()
    configs = _load_grid(grid_yaml)
    logger.info(
        "Running %d retrieval configs over %d queries",
        len(configs),
        len(golden.queries),
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out_path = RUNS_DIR / f"retrieval_{ts}.csv"

    fieldnames = [
        "config_id", "top_k", "variant", "hybrid", "rerank", "query_rewrite",
        "mean_precision_at_k", "mean_recall_at_k", "mean_mrr", "mean_ndcg_at_k",
        "latency_p50_ms", "cost_usd",
    ]

    rows = []
    for idx, cfg in enumerate(configs):
        config_id = f"ret_{idx:03d}"
        logger.info("Config %s: %s", config_id, cfg)
        metrics = _run_config(cfg, golden)
        row = {"config_id": config_id, **cfg, **metrics}
        rows.append(row)

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    best = max(rows, key=lambda r: r["mean_ndcg_at_k"])
    print(f"\nBest config by mean_ndcg_at_k ({best['mean_ndcg_at_k']:.4f}):")
    print(
        json.dumps(
            {
                k: best[k]
                for k in ["config_id", "top_k", "variant", "hybrid", "rerank", "query_rewrite"]
            },
            indent=2,
        )
    )
    BEST_PATH.write_text(json.dumps(best, indent=2))

    logger.info("Wrote %s", out_path)
    return out_path
