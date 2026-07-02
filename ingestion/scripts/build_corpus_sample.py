"""Build (and cache) the corpus id list for the full production ingest.

Discovers ~14k–17k TMDB movie ids across six vote-count / language / recency
tiers, deduplicates by tmdb_id (insertion order preserved via dict), and
caches the result to ``data/corpus_sample.json``.  A second run without
``--force`` reads the cache and returns immediately — no new TMDB calls.

Tier targets (approximate, TMDB catalogue changes over time):
  blockbuster  vote_count ≥ 5000                        ~1 026 films
  popular      vote_count 1000–4999                     ~3 849 films
  ok           vote_count 200–999                       ~9 737 films
  niche        Documentary/Western/War/Music/History/TV Movie
               vote_count 50–199 AND vote_average ≥ 6.5  ~600 films (capped)
  lang_pt_es   pt/es original language, vote_count ≥ 20
               AND vote_average ≥ 6.5                    ~300 films (capped)
  recent       primary_release_date ≥ today-90d,
               popularity.desc                           ~200 films (capped)

TMDB caps pages at 500 (20 results/page → 10 000 results per query); the
blockbuster and popular tiers stay well under that limit.  The ok tier
(~9 700 films) may need to be split by year range if TMDB ever truncates
it, but currently fits in ≤ 500 pages.

Usage:
    uv run python3 -m ingestion.scripts.build_corpus_sample
    uv run python3 -m ingestion.scripts.build_corpus_sample --force
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import date, timedelta
from pathlib import Path

from dotenv import load_dotenv

from ingestion.resources.tmdb_movies import TMDB_BASE, tmdb_get

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(message)s")
_logger = logging.getLogger(__name__)

CORPUS_CACHE = Path("data/corpus_sample.json")

# Genre ids for the niche-genre fill tier.
_NICHE_GENRE_IDS: list[int] = [
    99,     # Documentary
    37,     # Western
    10752,  # War
    10402,  # Music
    36,     # History
    10770,  # TV Movie
]

# TMDB hard cap on page numbers.
_TMDB_MAX_PAGE = 500


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _discover_tier(
    api_key: str,
    *,
    params: dict,
    cap: int | None = None,
    tier_name: str = "tier",
) -> list[int]:
    """Paginate /discover/movie with *params* until exhaustion or *cap* reached.

    Returns a list of tmdb ids (may contain duplicates across tiers — dedup
    happens in the caller).
    """
    ids: list[int] = []
    page = 1
    while True:
        resp = tmdb_get(
            f"{TMDB_BASE}/discover/movie",
            api_key=api_key,
            params={"page": page, **params},
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        total_pages = min(data.get("total_pages", 1), _TMDB_MAX_PAGE)

        for r in results:
            ids.append(r["id"])

        _logger.debug(
            '{"step":"tier_page","tier":"%s","page":%d,"total_pages":%d,"ids_so_far":%d}',
            tier_name, page, total_pages, len(ids),
        )

        if cap is not None and len(ids) >= cap:
            ids = ids[:cap]
            break
        if page >= total_pages:
            break
        page += 1

    _logger.info(
        '{"step":"tier_done","tier":"%s","count":%d}',
        tier_name, len(ids),
    )
    return ids


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_corpus(*, force: bool = False) -> list[int]:
    """Discover and cache the corpus id list; return the deduplicated ids."""
    if not force and CORPUS_CACHE.exists():
        cached = json.loads(CORPUS_CACHE.read_text())
        _logger.info(
            '{"step":"corpus_cached","ids":%d,"path":"%s"}',
            len(cached["tmdb_ids"]),
            CORPUS_CACHE,
        )
        return cached["tmdb_ids"]

    api_key = os.environ["TMDB_API_KEY"]

    # Ordered dict preserves insertion order and deduplicates.
    seen: dict[int, None] = {}

    def _merge(ids: list[int]) -> int:
        """Add ids to seen; return how many were new (first-seen tier attribution)."""
        before = len(seen)
        for i in ids:
            seen.setdefault(i, None)
        return len(seen) - before

    # --- Tier 1: blockbusters (vote_count ≥ 5000) ---
    blockbuster_ids = _discover_tier(
        api_key,
        params={
            "sort_by": "vote_count.desc",
            "vote_count.gte": 5000,
        },
        tier_name="blockbuster",
    )
    blockbuster_new = _merge(blockbuster_ids)

    # --- Tier 2: popular (vote_count 1000–4999) ---
    popular_ids = _discover_tier(
        api_key,
        params={
            "sort_by": "vote_count.desc",
            "vote_count.gte": 1000,
            "vote_count.lte": 4999,
        },
        tier_name="popular",
    )
    popular_new = _merge(popular_ids)

    # --- Tier 3: ok (vote_count 200–999) ---
    mid_ids = _discover_tier(
        api_key,
        params={
            "sort_by": "vote_count.desc",
            "vote_count.gte": 200,
            "vote_count.lte": 999,
        },
        tier_name="mid",
    )
    mid_new = _merge(mid_ids)

    # --- Tier 4: niche-genre fill (Documentary/Western/War/Music/History/TV Movie)
    #             vote_count 50–199 AND vote_average ≥ 6.5, capped ~600 total ---
    niche_cap_per_genre = 100  # 6 genres × 100 = 600
    niche_total: list[int] = []
    for genre_id in _NICHE_GENRE_IDS:
        genre_ids_tier = _discover_tier(
            api_key,
            params={
                "sort_by": "vote_average.desc",
                "with_genres": genre_id,
                "vote_count.gte": 50,
                "vote_count.lte": 199,
                "vote_average.gte": 6.5,
            },
            cap=niche_cap_per_genre,
            tier_name=f"niche_fill_{genre_id}",
        )
        niche_total.extend(genre_ids_tier)
    niche_new = _merge(niche_total)

    # --- Tier 5: pt/es original-language, vote_count ≥ 20, vote_average ≥ 6.5,
    #             capped ~300 (150 per language) ---
    lang_cap = 150
    lang_new = 0
    for lang in ("pt", "es"):
        lang_ids = _discover_tier(
            api_key,
            params={
                "sort_by": "vote_average.desc",
                "with_original_language": lang,
                "vote_count.gte": 20,
                "vote_average.gte": 6.5,
            },
            cap=lang_cap,
            tier_name=f"lang_{lang}",
        )
        lang_new += _merge(lang_ids)

    # --- Tier 6: recent releases (last 90 days), popularity.desc, capped ~200 ---
    date_gte = (date.today() - timedelta(days=90)).isoformat()
    recent_ids = _discover_tier(
        api_key,
        params={
            "sort_by": "popularity.desc",
            "primary_release_date.gte": date_gte,
        },
        cap=200,
        tier_name="recent",
    )
    recent_new = _merge(recent_ids)

    all_ids: list[int] = list(seen.keys())

    # --- Audit counts: new unique ids each tier contributed (first-seen wins),
    #     so the tier counts always sum to total_deduped. ---
    tier_counts = {
        "blockbuster": blockbuster_new,  # vote_count ≥ 5000
        "popular": popular_new,  # 1000–4999
        "mid": mid_new,  # 200–999
        "niche_fill": niche_new,  # thin-genre rescue, 50–199 & rating ≥ 6.5
        "regional_pt_es": lang_new,  # anglophone vote-bias correction
        "recent_90d": recent_new,  # vote-lag correction for new releases
        "total_deduped": len(all_ids),
    }

    CORPUS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    CORPUS_CACHE.write_text(
        json.dumps(
            {"tmdb_ids": all_ids, "tier_counts": tier_counts},
            indent=2,
        )
    )
    _logger.info(
        '{"step":"corpus_built","total":%d,"tier_counts":%s,"path":"%s"}',
        len(all_ids),
        json.dumps(tier_counts),
        CORPUS_CACHE,
    )
    return all_ids


def load_corpus() -> list[int]:
    """Return the cached corpus id list; error if it hasn't been built."""
    if not CORPUS_CACHE.exists():
        raise FileNotFoundError(
            f"{CORPUS_CACHE} missing — run "
            "`uv run python3 -m ingestion.scripts.build_corpus_sample` first"
        )
    return json.loads(CORPUS_CACHE.read_text())["tmdb_ids"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the corpus id list for the full ingest")
    parser.add_argument("--force", action="store_true", help="rebuild even if cached")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    build_corpus(force=args.force)
