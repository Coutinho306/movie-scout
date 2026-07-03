"""Assemble the 120-query DiagnosticSuite from the cached corpus golden set.

Loads ``data/golden_set_corpus_sample.json`` (the 30-film stratified sample),
fetches each target's full payload from ``tmdb_movies``, re-derives popularity
tier (same percentile cutoffs as ``golden_corpus_sample.py``) and review
coverage (``tmdb_reviews`` membership), builds tiers 0-2 mechanically and
copies tier 3 from the cached LLM text.

Usage::

    uv run python3 -m eval.diagnostic.build_suite          # use cache if present
    uv run python3 -m eval.diagnostic.build_suite --force  # rebuild
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from eval.golden import GoldenQuery, GoldenSet
from eval.diagnostic.tiers import (
    DiagnosticSuite,
    PopularityTier,
    ReviewCoverage,
    TierQuery,
    build_tier_queries,
)
from retrieval.client import get_qdrant_client

logger = logging.getLogger(__name__)

GOLDEN_CACHE = Path("data/golden_set_corpus_sample.json")
SUITE_CACHE = Path("data/diagnostic_suite.json")

_MOVIES_COLLECTION = "tmdb_movies"
_REVIEWS_COLLECTION = "tmdb_reviews"

# Percentile cutoffs — must mirror golden_corpus_sample.py exactly
_POPULAR_PERCENTILE = 0.95
_MID_LOW_PERCENTILE = 0.50
_MID_HIGH_PERCENTILE = 0.80
_NICHE_PERCENTILE = 0.30


def _fetch_review_covered_ids() -> set[int]:
    """Return distinct tmdb_ids that have at least one review chunk ingested."""
    client = get_qdrant_client()
    covered: set[int] = set()
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=_REVIEWS_COLLECTION,
            limit=1000,
            offset=offset,
            with_payload=["tmdb_id"],
            with_vectors=False,
        )
        for p in points:
            covered.add(p.payload["tmdb_id"])
        if offset is None:
            break
    logger.info('{"step":"review_covered_ids","count":%d}', len(covered))
    return covered


def _fetch_all_popularities() -> list[float]:
    """Fetch popularity values for all movies to derive percentile cutoffs."""
    client = get_qdrant_client()
    pops: list[float] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=_MOVIES_COLLECTION,
            limit=1000,
            offset=offset,
            with_payload=["popularity"],
            with_vectors=False,
        )
        for p in points:
            pops.append(p.payload.get("popularity", 0.0))
        if offset is None:
            break
    logger.info('{"step":"fetch_popularities","count":%d}', len(pops))
    return pops


def _derive_cutoffs(popularities: list[float]) -> tuple[float, float, float, float]:
    """Return (hi_cut, mid_lo_cut, mid_hi_cut, lo_cut) from the corpus percentiles."""
    pops = sorted(popularities)
    n = len(pops)
    hi_cut = pops[int(n * _POPULAR_PERCENTILE)]
    mid_lo_cut = pops[int(n * _MID_LOW_PERCENTILE)]
    mid_hi_cut = pops[int(n * _MID_HIGH_PERCENTILE)]
    lo_cut = pops[int(n * _NICHE_PERCENTILE)]
    return hi_cut, mid_lo_cut, mid_hi_cut, lo_cut


def _classify_popularity(
    popularity: float,
    *,
    hi_cut: float,
    mid_lo_cut: float,
    mid_hi_cut: float,
    lo_cut: float,
) -> PopularityTier:
    if popularity >= hi_cut:
        return "popular"
    if mid_lo_cut <= popularity < mid_hi_cut:
        return "mid"
    if popularity < lo_cut:
        return "niche"
    # values in [lo_cut, mid_lo_cut) or [mid_hi_cut, hi_cut) — assign nearest bucket
    if popularity < mid_lo_cut:
        return "niche"
    return "mid"


def _fetch_target_payload(tmdb_id: int) -> dict | None:
    """Fetch the full payload for a single target from tmdb_movies."""
    client = get_qdrant_client()
    results = client.scroll(
        collection_name=_MOVIES_COLLECTION,
        limit=1,
        scroll_filter={"must": [{"key": "tmdb_id", "match": {"value": tmdb_id}}]},
        with_payload=True,
        with_vectors=False,
    )
    points = results[0]
    if not points:
        return None
    return points[0].payload


def build_diagnostic_suite(force: bool = False) -> DiagnosticSuite:
    """Build (or load cached) the 120-query DiagnosticSuite.

    Re-derives popularity tier and review coverage from live corpus signals so
    the diagnostic labels stay in sync with the golden-set sampler's logic.
    """
    if not force and SUITE_CACHE.exists():
        logger.info("Loading diagnostic suite from cache: %s", SUITE_CACHE)
        data = json.loads(SUITE_CACHE.read_text())
        return DiagnosticSuite.model_validate(data)

    # Load cached golden set (30 films, tier-3 queries pre-generated)
    if not GOLDEN_CACHE.exists():
        raise FileNotFoundError(
            f"Golden corpus sample not found at {GOLDEN_CACHE}. "
            "Run eval/golden_corpus_sample.py first."
        )
    golden = GoldenSet.model_validate(json.loads(GOLDEN_CACHE.read_text()))
    logger.info('{"step":"golden_loaded","queries":%d}', len(golden.queries))

    # Derive corpus-wide popularity percentile cutoffs
    popularities = _fetch_all_popularities()
    hi_cut, mid_lo_cut, mid_hi_cut, lo_cut = _derive_cutoffs(popularities)
    logger.info(
        '{"step":"cutoffs","hi":%.2f,"mid_lo":%.2f,"mid_hi":%.2f,"lo":%.2f}',
        hi_cut, mid_lo_cut, mid_hi_cut, lo_cut,
    )

    # Collect tmdb_ids with review coverage
    review_covered = _fetch_review_covered_ids()

    all_queries: list[TierQuery] = []

    for gq in golden.queries:
        # Each GoldenQuery targets exactly one film in this golden set
        tmdb_id = next(iter(gq.target_tmdb_ids))
        tier3_text = gq.text

        payload = _fetch_target_payload(tmdb_id)
        if payload is None:
            logger.warning(
                '{"step":"missing_payload","tmdb_id":%d,"title":"%s"}',
                tmdb_id,
                gq.target_titles[0] if gq.target_titles else "unknown",
            )
            continue

        pop_val = payload.get("popularity", 0.0)
        popularity_tier: PopularityTier = _classify_popularity(
            pop_val,
            hi_cut=hi_cut,
            mid_lo_cut=mid_lo_cut,
            mid_hi_cut=mid_hi_cut,
            lo_cut=lo_cut,
        )
        review_coverage: ReviewCoverage = (
            "reviews" if tmdb_id in review_covered else "no_reviews"
        )

        tier_queries = build_tier_queries(
            payload,
            tmdb_id=tmdb_id,
            tier3_text=tier3_text,
            popularity_tier=popularity_tier,
            review_coverage=review_coverage,
        )
        all_queries.extend(tier_queries)

        logger.debug(
            '{"step":"film","tmdb_id":%d,"title":"%s","pop_tier":"%s","review":"%s"}',
            tmdb_id,
            payload.get("title", ""),
            popularity_tier,
            review_coverage,
        )

    suite = DiagnosticSuite(queries=all_queries)
    logger.info(
        '{"step":"suite_built","total_queries":%d,"films":%d}',
        len(suite.queries),
        len(suite.queries) // 4,
    )

    SUITE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    SUITE_CACHE.write_text(suite.model_dump_json())
    logger.info('{"step":"suite_cached","path":"%s"}', str(SUITE_CACHE))

    return suite


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description="Build the 120-query diagnostic suite from the corpus golden set"
    )
    parser.add_argument("--force", action="store_true", help="rebuild even if cached")
    args = parser.parse_args()
    suite = build_diagnostic_suite(force=args.force)
    print(f"Suite ready: {len(suite.queries)} queries across {len(suite.queries)//4} films")
