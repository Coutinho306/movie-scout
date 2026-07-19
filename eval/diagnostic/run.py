"""Diagnostic runner: evaluate the 120-query suite across all config variants.

Metrics per (tier × config) and per (tier × config × composition cell):
- nDCG@10, recall@10, recall@50, MRR@10

Verdict logic (AC6) and composition-attribution (AC7) are pure functions over
the recorded metrics.

Output:
- Timestamped CSV to ``eval/runs/diagnostic_<timestamp>.csv``
- 4×5 nDCG@10 plaintext summary to stdout and ``eval/runs/diagnostic_<timestamp>_summary.txt``

Usage::

    uv run python3 -m eval.diagnostic.run
"""

from __future__ import annotations

import csv
import json
import logging
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

from dotenv import load_dotenv

load_dotenv()

from agent.tools.query_mode import classify_query_mode
from eval.diagnostic.build_suite import build_diagnostic_suite
from eval.diagnostic.configs import CONFIGS, DiagnosticConfig
from eval.diagnostic.tiers import DiagnosticSuite
from eval.metrics.retrieval import mrr, ndcg_at_k, recall_at_k
from ingestion.config import Settings as IngestionSettings
from retrieval.config import RetrievalSettings
from retrieval.movies import search_movies
from retrieval.rerank import cross_encode_rerank

logger = logging.getLogger(__name__)

# Decision thresholds (tunable)
EPS_FLOOR = 0.02         # recall@10 gain below this is "no improvement"
REVIEW_MARGIN = 0.05     # review-attribution delta threshold


# ---------------------------------------------------------------------------
# Metric row key
# ---------------------------------------------------------------------------

class MetricKey(NamedTuple):
    config: str
    tier: int
    pop_tier: str   # "popular" | "mid" | "niche" | "ALL"
    review: str     # "reviews" | "no_reviews" | "ALL"


# ---------------------------------------------------------------------------
# Retrieval helpers
# ---------------------------------------------------------------------------

def _run_query(
    query_text: str,
    cfg: DiagnosticConfig,
    settings: RetrievalSettings,
) -> tuple[list[int], list[int]]:
    """Return (top10_ids, top50_ids) for a single query under a config.

    For non-rerank configs we issue one retrieval at k=50 and slice to 10.
    For rerank configs we fetch prefetch_k, rerank, then slice to 10; the
    pre-rerank pool provides the @50 ceiling.

    When ``cfg.route_hybrid=True``, the per-query hybrid flag is derived from
    ``classify_query_mode(query_text)`` rather than the fixed
    ``settings_kwargs["hybrid"]``. ``model_copy`` preserves the ingestion
    override (if any) so a pinned-variant run still targets the right collection.
    """
    if cfg.route_hybrid:
        effective_hybrid = classify_query_mode(query_text)
        effective_settings = settings.model_copy(
            update={"hybrid": effective_hybrid}
        )
    else:
        effective_settings = settings

    if cfg.rerank:
        pool = search_movies(query_text, settings=effective_settings, k=cfg.prefetch_k)
        top50_ids = [h.tmdb_id for h in pool[:50]]
        reranked = cross_encode_rerank(query_text, pool)
        top10_ids = [h.tmdb_id for h in reranked[:10]]
    else:
        hits50 = search_movies(query_text, settings=effective_settings, k=50)
        top50_ids = [h.tmdb_id for h in hits50]
        top10_ids = top50_ids[:10]

    return top10_ids, top50_ids


def _make_settings(
    cfg: DiagnosticConfig,
    ingestion: IngestionSettings | None = None,
) -> RetrievalSettings:
    """Build RetrievalSettings for a config, optionally pinning an ingestion variant.

    When ``ingestion`` is provided, the settings are pinned to that variant's
    collection and vector space via ``with_ingestion()``, so ``search_movies``
    queries the correct (e.g. ``calib_``-prefixed) collection instead of the
    production defaults read from ``.env``.
    """
    rs = RetrievalSettings(**cfg.settings_kwargs)
    if ingestion is not None:
        rs = rs.with_ingestion(ingestion)
    return rs


def _set_hyde_env(cfg: DiagnosticConfig) -> None:
    """Configure HYDE_BLEND_ALPHA in env before calling search_movies."""
    if not cfg.settings_kwargs.get("query_rewrite", False):
        # Not a HyDE config; env var irrelevant
        return
    if cfg.hyde_blend_alpha is not None:
        os.environ["HYDE_BLEND_ALPHA"] = str(cfg.hyde_blend_alpha)
    else:
        os.environ.pop("HYDE_BLEND_ALPHA", None)


def _restore_hyde_env(original: str | None) -> None:
    if original is None:
        os.environ.pop("HYDE_BLEND_ALPHA", None)
    else:
        os.environ["HYDE_BLEND_ALPHA"] = original


# ---------------------------------------------------------------------------
# Metrics accumulation
# ---------------------------------------------------------------------------

class _Accumulator:
    """Collects (ndcg10, recall10, recall50, mrr10) per MetricKey."""

    def __init__(self) -> None:
        self._data: dict[MetricKey, list[tuple[float, float, float, float]]] = defaultdict(list)

    def add(
        self,
        config: str,
        tier: int,
        pop_tier: str,
        review: str,
        top10: list[int],
        top50: list[int],
        relevant: set[int],
    ) -> None:
        n = ndcg_at_k(top10, relevant, 10)
        r10 = recall_at_k(top10, relevant, 10)
        r50 = recall_at_k(top50, relevant, 50)
        m10 = mrr(top10, relevant, 10)
        quad = (n, r10, r50, m10)
        # aggregate key (ALL × ALL)
        self._data[MetricKey(config, tier, "ALL", "ALL")].append(quad)
        # per-pop key
        self._data[MetricKey(config, tier, pop_tier, "ALL")].append(quad)
        # per-review key
        self._data[MetricKey(config, tier, "ALL", review)].append(quad)
        # full cell
        self._data[MetricKey(config, tier, pop_tier, review)].append(quad)

    def mean(self, key: MetricKey) -> tuple[float, float, float, float] | None:
        vals = self._data.get(key)
        if not vals:
            return None
        n = len(vals)
        return tuple(sum(v[i] for v in vals) / n for i in range(4))  # type: ignore[return-value]

    def count(self, key: MetricKey) -> int:
        return len(self._data.get(key, []))


# ---------------------------------------------------------------------------
# Verdict logic (AC6)
# ---------------------------------------------------------------------------

def derive_verdicts(
    acc: _Accumulator,
    config_names: list[str],
    baseline: str = "baseline_dense",
) -> dict[str, dict[int, dict]]:
    """Return per-config, per-tier verdict dicts.

    Structure: ``{config_name: {tier: {"verdict": str, "delta_recall10": float}}}``
    """
    result: dict[str, dict[int, dict]] = {}
    for cfg_name in config_names:
        if cfg_name == baseline:
            continue
        tier_verdicts: dict[int, dict] = {}
        # Compute deltas for tiers 0 and 1 to determine floor improvement
        deltas: dict[int, float] = {}
        for t in range(4):
            base_row = acc.mean(MetricKey(baseline, t, "ALL", "ALL"))
            cfg_row = acc.mean(MetricKey(cfg_name, t, "ALL", "ALL"))
            if base_row is None or cfg_row is None:
                deltas[t] = 0.0
            else:
                deltas[t] = cfg_row[1] - base_row[1]  # recall10 delta

        floor_improved = deltas[0] > EPS_FLOOR or deltas[1] > EPS_FLOOR

        if not floor_improved:
            for t in range(4):
                tier_verdicts[t] = {
                    "verdict": "not-attacking-root-cause" if t <= 1 else "uninformative",
                    "delta_recall10": round(deltas[t], 4),
                }
        else:
            # Find lowest tier where delta drops below EPS
            benefit_runs_out: int | None = None
            for t in range(4):
                if deltas[t] <= EPS_FLOOR:
                    benefit_runs_out = t
                    break
            for t in range(4):
                if benefit_runs_out is not None and t >= benefit_runs_out:
                    v = f"benefit-runs-out-at-tier-{benefit_runs_out}"
                else:
                    v = "improved"
                tier_verdicts[t] = {
                    "verdict": v,
                    "delta_recall10": round(deltas[t], 4),
                }

        result[cfg_name] = tier_verdicts
    return result


# ---------------------------------------------------------------------------
# Composition attribution (AC7)
# ---------------------------------------------------------------------------

def derive_attribution(
    acc: _Accumulator,
    config_names: list[str],
    baseline: str = "baseline_dense",
) -> dict[str, dict]:
    """Return per-config composition attribution.

    Structure::

        {config_name: {
            "impr_no_reviews": float,
            "impr_reviews": float,
            "reviews_vs_no_reviews_delta": float,
            "separate_review_work_indicated": bool,
        }}
    """
    result: dict[str, dict] = {}
    for cfg_name in config_names:
        if cfg_name == baseline:
            continue
        gains_no_rev: list[float] = []
        gains_rev: list[float] = []
        for pop_tier in ("popular", "mid", "niche"):
            for rev in ("reviews", "no_reviews"):
                for t in range(4):
                    k = MetricKey(cfg_name, t, pop_tier, rev)
                    bk = MetricKey(baseline, t, pop_tier, rev)
                    cfg_row = acc.mean(k)
                    base_row = acc.mean(bk)
                    if cfg_row is None or base_row is None:
                        continue
                    gain = cfg_row[1] - base_row[1]  # recall10 gain
                    if rev == "reviews":
                        gains_rev.append(gain)
                    else:
                        gains_no_rev.append(gain)

        impr_nr = sum(gains_no_rev) / len(gains_no_rev) if gains_no_rev else 0.0
        impr_r = sum(gains_rev) / len(gains_rev) if gains_rev else 0.0
        delta = impr_nr - impr_r
        result[cfg_name] = {
            "impr_no_reviews": round(impr_nr, 4),
            "impr_reviews": round(impr_r, 4),
            "reviews_vs_no_reviews_delta": round(delta, 4),
            "separate_review_work_indicated": delta > REVIEW_MARGIN,
        }
    return result


# ---------------------------------------------------------------------------
# CSV + summary output
# ---------------------------------------------------------------------------

_POP_TIERS = ("popular", "mid", "niche", "ALL")
_REVIEW_TIERS = ("reviews", "no_reviews", "ALL")
_TIERS = (0, 1, 2, 3)
_TIER_LABELS = {0: "title", 1: "overview", 2: "genre_mood", 3: "abstract"}


def _write_csv(
    acc: _Accumulator,
    config_names: list[str],
    out_path: Path,
) -> None:
    """Write per-(tier × pop × review) × config metrics to CSV."""
    metric_cols = ["ndcg10", "recall10", "recall50", "mrr10"]
    header = ["tier", "pop_tier", "review_coverage", "n"]
    for cfg in config_names:
        for m in metric_cols:
            header.append(f"{cfg}_{m}")

    rows = []
    for t in _TIERS:
        for pop in _POP_TIERS:
            for rev in _REVIEW_TIERS:
                row: dict = {
                    "tier": t,
                    "pop_tier": pop,
                    "review_coverage": rev,
                    "n": acc.count(MetricKey(config_names[0], t, pop, rev)),
                }
                for cfg in config_names:
                    means = acc.mean(MetricKey(cfg, t, pop, rev))
                    for i, m in enumerate(metric_cols):
                        row[f"{cfg}_{m}"] = f"{means[i]:.4f}" if means else "n=0"
                rows.append(row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    logger.info('{"step":"csv_written","path":"%s"}', str(out_path))


def _build_summary(
    acc: _Accumulator,
    config_names: list[str],
) -> str:
    """Build 4×5 nDCG@10 plaintext summary table."""
    col_w = 16
    tier_w = 20
    lines: list[str] = []
    lines.append("nDCG@10 Summary (tier rows × config columns)")
    lines.append("-" * (tier_w + col_w * len(config_names) + 4))

    # Header
    hdr = f"{'Tier':<{tier_w}}" + "".join(f"{c:>{col_w}}" for c in config_names)
    lines.append(hdr)
    lines.append("-" * (tier_w + col_w * len(config_names) + 4))

    for t in _TIERS:
        label = f"{t}:{_TIER_LABELS[t]}"
        cells = []
        for cfg in config_names:
            means = acc.mean(MetricKey(cfg, t, "ALL", "ALL"))
            cells.append(f"{means[0]:.4f}" if means else "  n/a ")
        row = f"{label:<{tier_w}}" + "".join(f"{c:>{col_w}}" for c in cells)
        lines.append(row)

    lines.append("-" * (tier_w + col_w * len(config_names) + 4))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Per-tier routing-rate report (AC-4)
# ---------------------------------------------------------------------------

def routing_rate_report(suite: DiagnosticSuite | None = None) -> str:
    """Return a formatted per-tier routing-rate report (no network, no retrieval).

    Runs classify_query_mode over every query in the suite and reports the
    fraction of each tier routed to hybrid. Pure classifier over cached suite
    text — does not call search_movies or any external service.
    """
    if suite is None:
        suite = build_diagnostic_suite()

    tier_total: dict[int, int] = {t: 0 for t in _TIERS}
    tier_hybrid: dict[int, int] = {t: 0 for t in _TIERS}

    for tq in suite.queries:
        tier_total[tq.tier] += 1
        if classify_query_mode(tq.text):
            tier_hybrid[tq.tier] += 1

    lines: list[str] = []
    lines.append("Per-tier hybrid-routing rate (classify_query_mode over suite text)")
    lines.append("-" * 60)
    lines.append(f"{'Tier':<20}{'hybrid':>10}{'total':>10}{'rate':>10}")
    lines.append("-" * 60)
    for t in _TIERS:
        h = tier_hybrid[t]
        total = tier_total[t]
        rate = h / total if total else 0.0
        label = f"{t}:{_TIER_LABELS[t]}"
        lines.append(f"{label:<20}{h:>10}{total:>10}{rate:>9.1%}")
    lines.append("-" * 60)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run(
    suite: DiagnosticSuite | None = None,
    ingestion: IngestionSettings | None = None,
) -> None:
    """Run the diagnostic suite across all config variants.

    Parameters
    ----------
    suite:
        Pre-built DiagnosticSuite.  When ``None``, built from the cache.
    ingestion:
        Optional ingestion variant to pin for all retrieval calls.  When
        provided, ``search_movies`` queries the variant collection (e.g. a
        ``calib_``-prefixed themes collection) instead of the production
        default.  ``None`` preserves existing prod behaviour.
    """
    if suite is None:
        suite = build_diagnostic_suite()

    config_names = [c.name for c in CONFIGS]
    acc = _Accumulator()
    original_hyde = os.environ.get("HYDE_BLEND_ALPHA")

    for cfg in CONFIGS:
        logger.info('{"step":"config_start","name":"%s"}', cfg.name)
        settings = _make_settings(cfg, ingestion)
        _set_hyde_env(cfg)
        try:
            for tq in suite.queries:
                top10, top50 = _run_query(tq.text, cfg, settings)
                relevant = {tq.target_tmdb_id}
                acc.add(
                    config=cfg.name,
                    tier=tq.tier,
                    pop_tier=tq.popularity_tier,
                    review=tq.review_coverage,
                    top10=top10,
                    top50=top50,
                    relevant=relevant,
                )
        finally:
            _restore_hyde_env(original_hyde)
        logger.info('{"step":"config_done","name":"%s"}', cfg.name)

    # Verdict block
    verdicts = derive_verdicts(acc, config_names)
    attribution = derive_attribution(acc, config_names)

    # Timestamped output
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    runs_dir = Path("eval/runs")
    csv_path = runs_dir / f"diagnostic_{ts}.csv"
    summary_path = runs_dir / f"diagnostic_{ts}_summary.txt"

    _write_csv(acc, config_names, csv_path)

    summary = _build_summary(acc, config_names)
    summary_path.write_text(summary + "\n")
    logger.info('{"step":"summary_written","path":"%s"}', str(summary_path))

    # Print summary
    print("\n" + summary + "\n")

    # Print recall@10 per tier per config
    print("recall@10 per tier × config:")
    hdr = f"{'Tier':<20}" + "".join(f"{c:>16}" for c in config_names)
    print(hdr)
    for t in _TIERS:
        label = f"{t}:{_TIER_LABELS[t]}"
        cells = []
        for cfg_name in config_names:
            means = acc.mean(MetricKey(cfg_name, t, "ALL", "ALL"))
            cells.append(f"{means[1]:.4f}" if means else "  n/a ")
        print(f"{label:<20}" + "".join(f"{c:>16}" for c in cells))
    print()

    # Print MRR@10
    print("MRR@10 per tier × config:")
    print(hdr)
    for t in _TIERS:
        label = f"{t}:{_TIER_LABELS[t]}"
        cells = []
        for cfg_name in config_names:
            means = acc.mean(MetricKey(cfg_name, t, "ALL", "ALL"))
            cells.append(f"{means[3]:.4f}" if means else "  n/a ")
        print(f"{label:<20}" + "".join(f"{c:>16}" for c in cells))
    print()

    # Print recall@50
    print("recall@50 per tier × config:")
    print(hdr)
    for t in _TIERS:
        label = f"{t}:{_TIER_LABELS[t]}"
        cells = []
        for cfg_name in config_names:
            means = acc.mean(MetricKey(cfg_name, t, "ALL", "ALL"))
            cells.append(f"{means[2]:.4f}" if means else "  n/a ")
        print(f"{label:<20}" + "".join(f"{c:>16}" for c in cells))
    print()

    # Print verdict block
    print("Verdict block (vs baseline_dense):")
    for cfg_name, tier_verdicts in verdicts.items():
        print(f"  {cfg_name}:")
        for t, vd in tier_verdicts.items():
            print(f"    tier{t}: {vd['verdict']:40s} Δrecall10={vd['delta_recall10']:+.4f}")
    print()

    # Print composition attribution
    print("Composition attribution (recall@10 gain: no_reviews vs reviews):")
    for cfg_name, attr in attribution.items():
        print(
            f"  {cfg_name}: impr_no_reviews={attr['impr_no_reviews']:+.4f}  "
            f"impr_reviews={attr['impr_reviews']:+.4f}  "
            f"delta={attr['reviews_vs_no_reviews_delta']:+.4f}  "
            f"separate_review_work_indicated={attr['separate_review_work_indicated']}"
        )
    print()

    # Save full results JSON alongside CSV
    results_json = {
        "timestamp": ts,
        "configs": config_names,
        "verdicts": {
            cfg: {str(t): v for t, v in tv.items()}
            for cfg, tv in verdicts.items()
        },
        "attribution": attribution,
    }
    results_path = runs_dir / f"diagnostic_{ts}_results.json"
    results_path.write_text(json.dumps(results_json, indent=2))
    logger.info('{"step":"results_json_written","path":"%s"}', str(results_path))

    print(f"CSV: {csv_path}")
    print(f"Summary: {summary_path}")
    print(f"Results JSON: {results_path}")


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Run the tiered diagnostic suite across all config variants"
    )
    parser.add_argument(
        "--variant",
        default=None,
        help=(
            "Ingestion variant suffix to evaluate against "
            "(e.g. 'calib_3small_c300o50_themes'). "
            "Defaults to production collections when omitted."
        ),
    )
    parser.add_argument(
        "--routing-report",
        action="store_true",
        default=False,
        help=(
            "Print the per-tier hybrid-routing rate from classify_query_mode "
            "over the 120-query suite text, then exit. No retrieval, no network."
        ),
    )
    cli_args = parser.parse_args()

    if cli_args.routing_report:
        print(routing_rate_report())
    else:
        ingestion_cfg: IngestionSettings | None = None
        if cli_args.variant:
            ingestion_cfg = IngestionSettings.from_variant_suffix(cli_args.variant)
        run(ingestion=ingestion_cfg)
