"""One-off backfill: re-process films whose theme extraction failed during the
production themes re-ingest (specs/features/0008-themes-production-reingest),
mostly due to a sustained OpenAI 429 rate-limit window under 8-worker load.

Re-runs the same fetch -> embed(themes) -> upsert path as the main loader
(_process_movie) for a fixed list of tmdb_ids, with a small worker pool
(default 3, well under the 8 that caused the original storm) now that
retry-with-backoff is in place in ingestion.theme_extraction.extract_themes.
The on-disk theme cache (keyed by tmdb_id) makes re-runs cheap: cached ids
skip the LLM call entirely and only re-embed + re-upsert.

Usage:
    uv run python3 -m ingestion.scripts.backfill_themes --ids-file <path>
    uv run python3 -m ingestion.scripts.backfill_themes --ids 123,456,789
    uv run python3 -m ingestion.scripts.backfill_themes --ids-file <path> --workers 3
"""

from __future__ import annotations

import argparse
import logging
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from qdrant_client.models import PointStruct

from ingestion.config import Settings
from ingestion.embedding import get_embedder
from ingestion.pipeline import get_qdrant_client
from ingestion.resources.tmdb_movies import fetch_movie_metadata

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(message)s")
_logger = logging.getLogger(__name__)
_log_lock = threading.Lock()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ids-file", help="path to a file with one tmdb_id per line")
    group.add_argument("--ids", help="comma-separated tmdb_ids")
    parser.add_argument(
        "--workers", type=int, default=3, help="concurrent workers (default 3)"
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.ids_file:
        with open(args.ids_file) as f:
            tmdb_ids = [int(line.strip()) for line in f if line.strip()]
    else:
        tmdb_ids = [int(x) for x in args.ids.split(",") if x.strip()]

    _logger.info(
        '{"step":"backfill_start","count":%d,"workers":%d}', len(tmdb_ids), args.workers
    )

    settings = Settings()
    embedder = get_embedder(settings)
    client = get_qdrant_client(os.environ["QDRANT_URL"], os.environ["QDRANT_API_KEY"])
    tmdb_api_key = os.environ["TMDB_API_KEY"]

    def _process(tmdb_id: int) -> str:
        try:
            metadata = fetch_movie_metadata(
                tmdb_id, tmdb_api_key, embed_text_recipe=settings.embed_text_recipe
            )
        except Exception as exc:
            _logger.warning(
                '{"step":"backfill_movie_error","tmdb_id":%d,"error":"%s"}',
                tmdb_id,
                type(exc).__name__,
            )
            return "failed"

        if metadata is None:
            _logger.warning('{"step":"backfill_fetch_miss","tmdb_id":%d}', tmdb_id)
            return "failed"

        if " Themes: " not in metadata.embed_text:
            _logger.warning('{"step":"backfill_still_empty","tmdb_id":%d}', tmdb_id)
            return "still_empty"

        dense_vector = embedder.embed_texts([metadata.embed_text])[0]
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(tmdb_id)))
        client.upsert(
            collection_name=settings.movies_collection,
            points=[
                PointStruct(
                    id=point_id,
                    vector=dense_vector,
                    payload={
                        "tmdb_id": metadata.tmdb_id,
                        "title": metadata.title,
                        "year": metadata.year,
                        "genres": metadata.genres,
                        "cast": metadata.cast,
                        "director": metadata.director,
                        "overview": metadata.overview,
                        "tagline": metadata.tagline,
                        "runtime": metadata.runtime,
                        "vote_average": metadata.vote_average,
                        "popularity": metadata.popularity,
                        "themes": metadata.themes,
                    },
                )
            ],
        )
        return "succeeded"

    succeeded = 0
    still_empty = 0
    failed = 0
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_id = {executor.submit(_process, tid): tid for tid in tmdb_ids}
        for future in as_completed(future_to_id):
            tmdb_id = future_to_id[future]
            result = future.result()
            with _log_lock:
                done += 1
                if result == "succeeded":
                    succeeded += 1
                elif result == "still_empty":
                    still_empty += 1
                else:
                    failed += 1
                _logger.info(
                    '{"step":"backfill_progress","done":%d,"total":%d,"tmdb_id":%d}',
                    done,
                    len(tmdb_ids),
                    tmdb_id,
                )

    _logger.info(
        '{"step":"backfill_complete","succeeded":%d,"still_empty":%d,"failed":%d}',
        succeeded,
        still_empty,
        failed,
    )


if __name__ == "__main__":
    main()
