"""Backfill the ``cast`` payload of every movie point to the top-15 credited actors.

Run via:
    uv run python3 scripts/backfill_cast_payload.py

Behaviour:
- Scrolls the movies collection in pages to enumerate all points.
- Skip predicate (resumable): if a point's stored ``cast`` already has ≥15
  entries, skip it — no TMDB call, no write.  This makes the script naturally
  resumable after a crash: the corpus state itself is the checkpoint.
- For each remaining point: TMDB GET /movie/{tmdb_id}?append_to_response=credits,
  take cast[:15] names.
- Write payload-only via client.set_payload (NOT overwrite_payload — that would
  replace the whole payload dict; NOT upsert — that requires the vector too).
  set_payload merges only the ``cast`` key onto the existing point.
- Per-movie try/except so a single TMDB miss does not kill the whole run.
- Periodic progress logging every PROGRESS_EVERY points.
"""

from __future__ import annotations

import logging
import os
import time

from dotenv import load_dotenv
from qdrant_client import QdrantClient

from ingestion.resources.tmdb_movies import TMDB_BASE, tmdb_get

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(message)s")
_logger = logging.getLogger(__name__)

PROGRESS_EVERY = 200
_PAGE_SIZE = 100


def _fetch_cast_top15(tmdb_id: int, api_key: str) -> list[str] | None:
    """Fetch top-15 cast names for ``tmdb_id`` from TMDB credits.

    Returns None on any non-recoverable error (logged); the backfill loop
    treats None as a skip with failure count increment.
    """
    resp = tmdb_get(
        f"{TMDB_BASE}/movie/{tmdb_id}",
        api_key=api_key,
        params={"append_to_response": "credits"},
    )
    if resp.status_code != 200:
        _logger.warning(
            '{"step":"backfill_cast","tmdb_id":%d,"status":%d,"action":"skip"}',
            tmdb_id,
            resp.status_code,
        )
        return None
    data = resp.json()
    names = [c["name"] for c in data.get("credits", {}).get("cast", [])[:15]]
    return names


def backfill(
    *,
    qdrant_url: str,
    qdrant_api_key: str,
    tmdb_api_key: str,
    collection_name: str,
) -> None:
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=30)

    done = 0
    skipped = 0
    failed = 0
    total = 0

    next_offset = None

    while True:
        records, next_offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=None,
            limit=_PAGE_SIZE,
            offset=next_offset,
            with_payload=["tmdb_id", "cast"],
            with_vectors=False,
        )

        if not records:
            break

        for record in records:
            total += 1
            p = record.payload or {}
            stored_cast: list = p.get("cast") or []

            # Skip predicate: already at top-15 (or more) — resumability guarantee.
            if len(stored_cast) >= 15:
                skipped += 1
                if total % PROGRESS_EVERY == 0:
                    _logger.info(
                        '{"step":"backfill_progress","total":%d,"done":%d,"skipped":%d,"failed":%d}',
                        total, done, skipped, failed,
                    )
                continue

            tmdb_id: int = p.get("tmdb_id", 0)
            if not tmdb_id:
                failed += 1
                continue

            try:
                names = _fetch_cast_top15(tmdb_id, tmdb_api_key)
                if names is None:
                    failed += 1
                    continue

                # Payload-only merge: set_payload touches only "cast", not vectors.
                client.set_payload(
                    collection_name=collection_name,
                    payload={"cast": names},
                    points=[record.id],
                )
                done += 1

            except Exception as exc:  # noqa: BLE001
                _logger.warning(
                    '{"step":"backfill_cast","tmdb_id":%d,"error":"%s"}',
                    tmdb_id,
                    str(exc)[:120],
                )
                failed += 1

            if total % PROGRESS_EVERY == 0:
                _logger.info(
                    '{"step":"backfill_progress","total":%d,"done":%d,"skipped":%d,"failed":%d}',
                    total, done, skipped, failed,
                )

        if next_offset is None:
            break

    _logger.info(
        '{"step":"backfill_done","total":%d,"done":%d,"skipped":%d,"failed":%d}',
        total, done, skipped, failed,
    )


if __name__ == "__main__":
    qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key = os.environ.get("QDRANT_API_KEY", "")
    tmdb_api_key = os.environ["TMDB_API_KEY"]
    collection_name = os.environ.get("MOVIES_COLLECTION", "tmdb_movies")

    _logger.info(
        '{"step":"backfill_start","collection":"%s","qdrant_url":"%s"}',
        collection_name,
        qdrant_url,
    )
    backfill(
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
        tmdb_api_key=tmdb_api_key,
        collection_name=collection_name,
    )
    time.sleep(0)  # allow any pending log flushes
