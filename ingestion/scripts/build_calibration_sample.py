"""Build (and cache) the fixed calibration sample: golden targets + distractors.

The SAME id list must feed every embedder/chunk/recipe variant, or the variants
aren't comparable. So the sample is built once, cached to
``data/calibration_sample.json``, and reused across all variant ingests.

Composition (per the calibration spike):
- All golden targets (guarantees each query's answer is in-corpus).
- ~N distractors from TMDB discovery (keeps enough competition that nDCG isn't
  trivially inflated by a near-empty index).

Usage:
    uv run python3 -m ingestion.scripts.build_calibration_sample --distractors 300
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from eval.golden import build_golden_set
from ingestion.resources.tmdb_movies import DISCOVERY_GENRES, discover_candidate_tmdb_ids

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(message)s")
_logger = logging.getLogger(__name__)

SAMPLE_CACHE = Path("data/calibration_sample.json")


def build_sample(*, distractors: int, force: bool = False) -> list[int]:
    if not force and SAMPLE_CACHE.exists():
        cached = json.loads(SAMPLE_CACHE.read_text())
        _logger.info(
            '{"step":"sample_cached","ids":%d,"path":"%s"}',
            len(cached["tmdb_ids"]),
            SAMPLE_CACHE,
        )
        return cached["tmdb_ids"]

    golden = build_golden_set()
    target_ids = sorted(golden.holdout_tmdb_ids)

    api_key = os.environ["TMDB_API_KEY"]
    # discovery_pages sized so the pool comfortably exceeds `distractors`.
    pool = discover_candidate_tmdb_ids(api_key, pages=8, genre_ids=DISCOVERY_GENRES)
    target_set = set(target_ids)
    distractor_ids = [i for i in pool if i not in target_set][:distractors]

    sample_ids = target_ids + distractor_ids
    SAMPLE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    SAMPLE_CACHE.write_text(
        json.dumps(
            {
                "tmdb_ids": sample_ids,
                "target_count": len(target_ids),
                "distractor_count": len(distractor_ids),
            },
            indent=2,
        )
    )
    _logger.info(
        '{"step":"sample_built","targets":%d,"distractors":%d,"total":%d,"path":"%s"}',
        len(target_ids),
        len(distractor_ids),
        len(sample_ids),
        SAMPLE_CACHE,
    )
    return sample_ids


def load_sample() -> list[int]:
    """Return the cached sample id list; error if it hasn't been built."""
    if not SAMPLE_CACHE.exists():
        raise FileNotFoundError(
            f"{SAMPLE_CACHE} missing — run "
            "`uv run python3 -m ingestion.scripts.build_calibration_sample` first"
        )
    return json.loads(SAMPLE_CACHE.read_text())["tmdb_ids"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the fixed calibration sample")
    parser.add_argument("--distractors", type=int, default=300)
    parser.add_argument("--force", action="store_true", help="rebuild even if cached")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    build_sample(distractors=args.distractors, force=args.force)
