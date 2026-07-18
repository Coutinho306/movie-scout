"""Grid search over LLM knobs (temperature, prompt variant); writes CSV."""
from __future__ import annotations

import csv
import itertools
import json
import logging
import random
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from agent.config import AgentSettings
from agent.main import run as agent_run
from eval.golden import build_golden_set
from eval.metrics.llm import (
    hallucination_rate,
    ragas_answer_relevancy,
    ragas_faithfulness,
    taste_match,
)

logger = logging.getLogger(__name__)
RUNS_DIR = Path("eval/runs")
DEFAULT_GRID = Path("eval/grids/llm.yaml")
BEST_RETRIEVAL = Path("eval/runs/best_retrieval.json")


def _load_grid(grid_yaml: Path) -> tuple[list[dict], int]:
    with grid_yaml.open() as f:
        raw = yaml.safe_load(f)
    sample_queries = raw.pop("sample_queries", 10)
    keys = list(raw.keys())
    values = [raw[k] for k in keys]
    configs = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    return configs, sample_queries


def run(grid_yaml: Path = DEFAULT_GRID) -> Path:
    """Run LLM eval grid; return path to output CSV."""
    if not BEST_RETRIEVAL.exists():
        raise FileNotFoundError(
            f"{BEST_RETRIEVAL} not found. Run `eval.cli retrieval` first."
        )

    best_ret = json.loads(BEST_RETRIEVAL.read_text())
    logger.info("Pinned retrieval config: %s", best_ret.get("config_id"))

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    golden = build_golden_set()
    configs, n_sample = _load_grid(grid_yaml)

    rng = random.Random(42)
    sample = rng.sample(golden.queries, min(n_sample, len(golden.queries)))
    logger.info("Running %d LLM configs over %d queries", len(configs), len(sample))

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out_path = RUNS_DIR / f"llm_{ts}.csv"

    fieldnames = [
        "config_id", "temperature", "prompt_variant",
        "faithfulness", "answer_relevancy", "taste_match",
        "hallucination_rate", "latency_p50_ms", "cost_usd",
    ]

    rows = []
    for idx, cfg in enumerate(configs):
        config_id = f"llm_{idx:03d}"
        logger.info("LLM config %s: %s", config_id, cfg)

        f_vals, ar_vals, tm_vals, hr_vals = [], [], [], []
        latencies: list[float] = []
        total_cost = 0.0

        settings = AgentSettings(
            temperature=cfg["temperature"], prompt_variant=cfg["prompt_variant"]
        )

        for gq in sample:
            t0 = time.time()
            try:
                result = agent_run(gq.text, settings=settings)
            except Exception as exc:  # noqa: BLE001 — skip failed run, keep grid going
                logger.warning("agent_run failed: %s", exc)
                continue
            latency_ms = (time.time() - t0) * 1000
            latencies.append(latency_ms)
            total_cost += result.cost_usd

            # Contexts = top RAG hits titles (simplified)
            contexts = [c.title for c in result.citations]
            try:
                f_vals.append(ragas_faithfulness(gq.text, result.final_answer, contexts))
                ar_vals.append(ragas_answer_relevancy(gq.text, result.final_answer, contexts))
            except Exception as exc:  # noqa: BLE001 — one query's RAGAS failure shouldn't sink a paid grid run
                logger.error("RAGAS scoring failed for query %r: %s", gq.text, exc)
                f_vals.append(float("nan"))
                ar_vals.append(float("nan"))
            tm_vals.append(taste_match(result.final_answer))

            cited_ids = [c.tmdb_id for c in result.citations]
            hr_vals.append(hallucination_rate(cited_ids, result.retrieved_tmdb_ids))

        def safe_mean(vals: list) -> float:
            valid = [v for v in vals if v == v]  # filter nan
            return statistics.mean(valid) if valid else float("nan")

        latencies_sorted = sorted(latencies)
        p50 = latencies_sorted[len(latencies_sorted) // 2] if latencies_sorted else 0.0

        row = {
            "config_id": config_id,
            "temperature": cfg["temperature"],
            "prompt_variant": cfg["prompt_variant"],
            "faithfulness": round(safe_mean(f_vals), 4),
            "answer_relevancy": round(safe_mean(ar_vals), 4),
            "taste_match": round(safe_mean(tm_vals), 4),
            "hallucination_rate": round(safe_mean(hr_vals), 4),
            "latency_p50_ms": round(p50, 1),
            "cost_usd": round(total_cost, 6),
        }
        rows.append(row)

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if rows:
        best = max(rows, key=lambda r: r.get("answer_relevancy") or 0)
        print("\nBest LLM config by answer_relevancy:")
        print(
            json.dumps(
                {
                    k: best[k]
                    for k in ["config_id", "temperature", "prompt_variant", "answer_relevancy"]
                },
                indent=2,
            )
        )

    logger.info("Wrote %s", out_path)
    return out_path
