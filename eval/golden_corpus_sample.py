"""Ground truth (variant 2): sample real films directly from the ingested corpus,
stratified by popularity tier and review-chunk presence, instead of the personal
watchlist (`eval/golden.py`).

The watchlist-based golden set skews toward films the owner rates highly enough to
want to watch, which turned out to correlate with films TMDB has no user reviews
for (see specs/features/corpus-seed-snapshot/STATUS.md). This set samples directly
from `tmdb_movies`/`tmdb_reviews` so the eval reflects a realistic mix: popular /
mid-tier / niche films, roughly half with review-chunk coverage and half without.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic_settings import BaseSettings, SettingsConfigDict

from eval.golden import GoldenQuery, GoldenSet
from retrieval.client import get_qdrant_client

load_dotenv()
logger = logging.getLogger(__name__)

GOLDEN_CACHE = Path("data/golden_set_corpus_sample.json")
QUERY_GEN_PROMPT = Path(__file__).parent / "prompts/query_gen.md"

_MOVIES_COLLECTION = "tmdb_movies"
_REVIEWS_COLLECTION = "tmdb_reviews"

# Popularity-percentile cutoffs (computed from the live corpus, not hardcoded
# thresholds) split candidates into popular / mid / niche tiers.
_POPULAR_PERCENTILE = 0.95
_MID_LOW_PERCENTILE = 0.50
_MID_HIGH_PERCENTILE = 0.80
_NICHE_PERCENTILE = 0.30

_PER_TIER_WITH_REVIEWS = 5
_PER_TIER_WITHOUT_REVIEWS = 5
_MIN_VOTE_AVERAGE = 5.5

_SEED = 42


class _EvalSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    model_orchestrator: str = "gpt-4o-mini"


def _fetch_review_covered_ids() -> set[int]:
    """Distinct tmdb_ids that have at least one review chunk ingested."""
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
    return covered


def _fetch_all_movies() -> list[dict]:
    client = get_qdrant_client()
    movies: list[dict] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=_MOVIES_COLLECTION,
            limit=1000,
            offset=offset,
            with_payload=["tmdb_id", "title", "year", "popularity", "vote_average"],
            with_vectors=False,
        )
        movies.extend(p.payload for p in points)
        if offset is None:
            break
    return movies


def _stratify(movies: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    pops = sorted(m["popularity"] for m in movies)
    n = len(pops)
    hi_cut = pops[int(n * _POPULAR_PERCENTILE)]
    mid_lo_cut = pops[int(n * _MID_LOW_PERCENTILE)]
    mid_hi_cut = pops[int(n * _MID_HIGH_PERCENTILE)]
    lo_cut = pops[int(n * _NICHE_PERCENTILE)]

    popular = [m for m in movies if m["popularity"] >= hi_cut]
    mid = [m for m in movies if mid_lo_cut <= m["popularity"] < mid_hi_cut]
    niche = [m for m in movies if m["popularity"] < lo_cut]
    return popular, mid, niche


def _sample_tier(
    pool: list[dict],
    with_reviews: set[int],
    *,
    n_with: int,
    n_without: int,
    rng: random.Random,
) -> list[dict]:
    quality_pool = [m for m in pool if m.get("vote_average", 0) >= _MIN_VOTE_AVERAGE]
    has_reviews = [m for m in quality_pool if m["tmdb_id"] in with_reviews]
    no_reviews = [m for m in quality_pool if m["tmdb_id"] not in with_reviews]
    rng.shuffle(has_reviews)
    rng.shuffle(no_reviews)
    return has_reviews[:n_with] + no_reviews[:n_without]


def _sample_candidates() -> list[dict]:
    movies = _fetch_all_movies()
    with_reviews = _fetch_review_covered_ids()
    popular, mid, niche = _stratify(movies)

    rng = random.Random(_SEED)
    picked = (
        _sample_tier(
            popular,
            with_reviews,
            n_with=_PER_TIER_WITH_REVIEWS,
            n_without=_PER_TIER_WITHOUT_REVIEWS,
            rng=rng,
        )
        + _sample_tier(
            mid,
            with_reviews,
            n_with=_PER_TIER_WITH_REVIEWS,
            n_without=_PER_TIER_WITHOUT_REVIEWS,
            rng=rng,
        )
        + _sample_tier(
            niche,
            with_reviews,
            n_with=_PER_TIER_WITH_REVIEWS,
            n_without=_PER_TIER_WITHOUT_REVIEWS,
            rng=rng,
        )
    )
    return picked


def _generate_queries(candidates: list[dict], model: str) -> list[GoldenQuery]:
    template = QUERY_GEN_PROMPT.read_text()
    llm = ChatOpenAI(model=model, temperature=0.7)
    queries: list[GoldenQuery] = []
    for movie in candidates:
        prompt = template.format(title=movie["title"], year=movie["year"])
        response = llm.invoke(prompt)
        query_text = response.content.strip().strip('"')
        queries.append(
            GoldenQuery(
                text=query_text,
                target_tmdb_ids={movie["tmdb_id"]},
                target_titles=[movie["title"]],
            )
        )
    return queries


def build_golden_set_from_corpus(force: bool = False) -> GoldenSet:
    """Build (or load cached) a corpus-sampled GoldenSet.

    Stratified across popular/mid/niche popularity tiers and review-chunk
    presence, so nDCG measurements reflect the real corpus mix instead of a
    watchlist that happens to skew toward review-less films.
    """
    if not force and GOLDEN_CACHE.exists():
        logger.info("Loading corpus-sample golden set from cache: %s", GOLDEN_CACHE)
        data = json.loads(GOLDEN_CACHE.read_text())
        return GoldenSet.model_validate(data)

    settings = _EvalSettings()
    candidates = _sample_candidates()
    logger.info(
        '{"step":"golden_corpus_sample","candidates":%d}', len(candidates)
    )
    queries = _generate_queries(candidates, settings.model_orchestrator)

    golden = GoldenSet(
        holdout_tmdb_ids={m["tmdb_id"] for m in candidates},
        queries=queries,
    )
    GOLDEN_CACHE.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN_CACHE.write_text(golden.model_dump_json())
    logger.info(
        '{"step":"golden_corpus_sample_cached","path":"%s","queries":%d}',
        str(GOLDEN_CACHE),
        len(queries),
    )
    return golden


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description="Build the corpus-sampled golden set (30 queries, stratified)"
    )
    parser.add_argument("--force", action="store_true", help="rebuild even if cached")
    args = parser.parse_args()
    build_golden_set_from_corpus(force=args.force)
