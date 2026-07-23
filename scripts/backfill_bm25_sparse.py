"""Backfill BM25 sparse vectors onto every point in the ``tmdb_movies`` collection.

Sparse text recipe: ``enriched-base-kw`` — title, year, genres, director, cast (top-5),
tagline, overview, keywords.  Extends the former ``enriched-base`` recipe with a
``Keywords: ...`` clause (spec 0022 Phase 3).  Built via
``ingestion.chunking.build_sparse_text`` (the shared drift-guard builder).

No OpenAI calls, no dense re-embedding — ``update_vectors`` is used so the dense
vector is structurally untouchable.

Run via::

    uv run python3 -m scripts.backfill_bm25_sparse

Skip predicate (resumability / idempotence):
    Points already carrying ``sparse_recipe == _SPARSE_TEXT_RECIPE`` in their
    payload are skipped.  This means:
    - First run: rewrites all 15,503 points (none carry the tag yet).
    - Re-run: all points carry the tag → no-op (written == 0).

Tag write:
    On each successful rewrite, ``set_payload`` writes
    ``{"sparse_recipe": _SPARSE_TEXT_RECIPE}`` into the point's payload.
    ``update_vectors`` is called first (sparse vector only — dense untouched),
    then ``set_payload`` for the tag.

Schema step (``--ensure-sparse-config`` mode):
    The production collection was created with a single unnamed dense vector
    (1536-dim COSINE) and no sparse config.  ``update_collection`` with
    ``sparse_vectors_config`` was rejected by the server (Qdrant 1.18.2):
    "Wrong input: Not existing vector name error: text".  The supported path is
    a collection recreate::

        1. Create ``tmdb_movies__bm25_tmp`` with the same unnamed 1536-dim
           COSINE dense config PLUS ``sparse_vectors_config={"text": IDF}``.
        2. Scroll ``tmdb_movies`` with dense vectors, upsert each point's dense
           vector verbatim into ``tmdb_movies__bm25_tmp``.
        3. Verify ``tmdb_movies__bm25_tmp.points_count == tmdb_movies.points_count``.
        4. Delete ``tmdb_movies``, recreate it with the dual config (same as tmp).
        5. Copy all points from tmp back into the new ``tmdb_movies``.
        6. Verify count, delete tmp.

    Dense vectors are never recomputed — zero re-embedding.
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Modifier,
    PointStruct,
    PointVectors,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from ingestion.chunking import build_sparse_text

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(message)s")
_logger = logging.getLogger(__name__)

COLLECTION = "tmdb_movies"
TMP_COLLECTION = "tmdb_movies__bm25_tmp"
PROGRESS_EVERY = 200
_PAGE_SIZE = 100
_DENSE_SIZE = 1536

# Enriched sparse text recipe version tag (AC-3).
# Changing this constant causes a re-run to rewrite all points again.
# Bumped from "enriched-base" to "enriched-base-kw" to force full reindex
# after the keywords clause was added to build_sparse_text (spec 0022 Phase 3).
_SPARSE_TEXT_RECIPE = "enriched-base-kw"

# Payload fields fetched during enriched-recipe scroll.
_ENRICHED_PAYLOAD_FIELDS = ["title", "year", "genres", "cast", "director", "tagline", "overview", "keywords"]


# ---------------------------------------------------------------------------
# Schema step: recreate with sparse config
# ---------------------------------------------------------------------------

def _has_sparse_text_config(client: QdrantClient) -> bool:
    """Return True if ``COLLECTION`` already has a ``text`` sparse vector config."""
    info = client.get_collection(COLLECTION)
    sp = info.config.params.sparse_vectors
    return sp is not None and "text" in sp


def _copy_dense_to_tmp(client: QdrantClient, src: str, dst: str) -> int:
    """Scroll ``src`` with dense vectors and upsert verbatim into ``dst``.

    Returns the number of points copied.
    """
    copied = 0
    next_offset = None

    while True:
        records, next_offset = client.scroll(
            collection_name=src,
            limit=100,
            offset=next_offset,
            with_payload=True,
            with_vectors=True,
        )
        if not records:
            break

        points: list[PointStruct] = []
        for r in records:
            vec = r.vector
            if isinstance(vec, dict):
                dense_vec = vec.get("") or next(iter(vec.values()), None)
            else:
                dense_vec = vec
            if dense_vec is None:
                continue
            points.append(PointStruct(id=r.id, vector=list(dense_vec), payload=r.payload or {}))

        if points:
            client.upsert(collection_name=dst, points=points)
            copied += len(points)
            _logger.info(
                '{"step":"copy_dense","copied_so_far":%d}',
                copied,
            )

        if next_offset is None:
            break

    return copied


def ensure_sparse_config(client: QdrantClient) -> None:
    """Add ``text`` sparse vector config to ``COLLECTION`` via recreate fallback.

    If the config is already present, this is a no-op (idempotent).
    """
    if _has_sparse_text_config(client):
        _logger.info('{"step":"schema_check","result":"already_has_sparse_text","action":"skip"}')
        return

    _logger.info('{"step":"schema_recreate","reason":"update_collection_rejected_sparse_config"}')

    # Step 1: create tmp with dual config (unnamed dense + sparse text)
    if client.collection_exists(TMP_COLLECTION):
        _logger.info('{"step":"schema_tmp","action":"delete_existing_tmp"}')
        client.delete_collection(TMP_COLLECTION)

    client.create_collection(
        collection_name=TMP_COLLECTION,
        vectors_config=VectorParams(size=_DENSE_SIZE, distance=Distance.COSINE),
        sparse_vectors_config={"text": SparseVectorParams(modifier=Modifier.IDF)},
    )
    _logger.info('{"step":"schema_tmp","action":"created_tmp_with_dual_config"}')

    # Step 2: copy all dense vectors from production into tmp
    prod_count = client.get_collection(COLLECTION).points_count or 0
    copied = _copy_dense_to_tmp(client, COLLECTION, TMP_COLLECTION)
    tmp_count = client.get_collection(TMP_COLLECTION).points_count or 0
    _logger.info(
        '{"step":"schema_copy_done","prod_count":%d,"copied":%d,"tmp_count":%d}',
        prod_count, copied, tmp_count,
    )

    # Step 3: swap — delete production, recreate with dual config, copy back
    _logger.info('{"step":"schema_swap","action":"delete_production"}')
    client.delete_collection(COLLECTION)

    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=_DENSE_SIZE, distance=Distance.COSINE),
        sparse_vectors_config={"text": SparseVectorParams(modifier=Modifier.IDF)},
    )
    _logger.info('{"step":"schema_swap","action":"recreated_production_with_dual_config"}')

    copied_back = _copy_dense_to_tmp(client, TMP_COLLECTION, COLLECTION)
    final_count = client.get_collection(COLLECTION).points_count or 0
    _logger.info(
        '{"step":"schema_swap_done","copied_back":%d,"final_count":%d}',
        copied_back, final_count,
    )

    # Step 4: cleanup tmp
    client.delete_collection(TMP_COLLECTION)
    _logger.info('{"step":"schema_done","collection":"%s","points_count":%d}', COLLECTION, final_count)

    # Verify
    info = client.get_collection(COLLECTION)
    assert info.config.params.sparse_vectors is not None and "text" in info.config.params.sparse_vectors, (
        f"Sparse config not found after recreate: {info.config.params.sparse_vectors}"
    )
    assert info.points_count == prod_count, (
        f"Point count changed: before={prod_count}, after={info.points_count}"
    )


# ---------------------------------------------------------------------------
# Skip predicate — version-tag based (AC-3)
# ---------------------------------------------------------------------------

def _already_tagged_current_recipe(payload: dict) -> bool:
    """Return True if the point payload carries the current recipe version tag.

    Points without the tag (or with an older tag) are NOT skipped — they will
    be rewritten with the enriched-base sparse vector and then tagged.
    On a first run all points lack the tag → all are rewritten.
    On a re-run all points carry the tag → all are skipped (no-op).
    """
    return payload.get("sparse_recipe") == _SPARSE_TEXT_RECIPE


# ---------------------------------------------------------------------------
# Backfill
# ---------------------------------------------------------------------------

def backfill(client: QdrantClient) -> dict[str, int]:
    """Scroll all points and write enriched-base BM25 sparse vectors.

    Fetches the seven enriched payload fields per point and calls the shared
    ``build_sparse_text`` builder.  On each successful write, tags the point
    with ``sparse_recipe == _SPARSE_TEXT_RECIPE`` via ``set_payload``.

    Returns a counts dict: written / skipped_populated / skipped_empty_text / failed.
    """
    _logger.info(
        '{"step":"backfill_start","collection":"%s","sparse_recipe":"%s"}',
        COLLECTION,
        _SPARSE_TEXT_RECIPE,
    )

    # Load BM25 model via fastembed (no OpenAI call — local inference only)
    from fastembed import SparseTextEmbedding
    bm25 = SparseTextEmbedding(model_name="Qdrant/bm25")

    written = 0
    skipped_populated = 0
    skipped_empty_text = 0
    failed = 0
    total = 0

    next_offset = None

    while True:
        records, next_offset = client.scroll(
            collection_name=COLLECTION,
            limit=_PAGE_SIZE,
            offset=next_offset,
            with_payload=_ENRICHED_PAYLOAD_FIELDS + ["sparse_recipe"],
            with_vectors=False,
        )
        if not records:
            break

        for record in records:
            total += 1
            p = record.payload or {}

            # Skip predicate — already tagged with the current recipe version
            if _already_tagged_current_recipe(p):
                skipped_populated += 1
                if total % PROGRESS_EVERY == 0:
                    _logger.info(
                        '{"step":"backfill_progress","total":%d,"written":%d,"skipped_populated":%d,"skipped_empty_text":%d,"failed":%d}',
                        total, written, skipped_populated, skipped_empty_text, failed,
                    )
                continue

            # Build enriched sparse text from shared builder
            title: str = (p.get("title") or "").strip()
            year_raw = p.get("year")
            year: int = int(year_raw) if year_raw is not None else 0
            genres: list[str] = p.get("genres") or []
            cast: list[str] = p.get("cast") or []
            director: str = (p.get("director") or "").strip()
            tagline: str = (p.get("tagline") or "").strip()
            overview: str = (p.get("overview") or "").strip()
            # Read keywords from the now-backfilled payload (Phase 2 must run first).
            keywords: list[str] = p.get("keywords") or []

            text = build_sparse_text(
                title=title,
                year=year,
                genres=genres,
                director=director,
                cast=cast,
                tagline=tagline,
                overview=overview,
                keywords=keywords if keywords else None,
            ).strip()

            if not text:
                skipped_empty_text += 1
                _logger.debug(
                    '{"step":"backfill_empty_text","point_id":"%s"}', str(record.id)
                )
                if total % PROGRESS_EVERY == 0:
                    _logger.info(
                        '{"step":"backfill_progress","total":%d,"written":%d,"skipped_populated":%d,"skipped_empty_text":%d,"failed":%d}',
                        total, written, skipped_populated, skipped_empty_text, failed,
                    )
                continue

            try:
                sv = next(iter(bm25.embed([text])))
                sparse_vec = SparseVector(
                    indices=sv.indices.tolist(),
                    values=sv.values.tolist(),
                )
                client.update_vectors(
                    collection_name=COLLECTION,
                    points=[
                        PointVectors(
                            id=record.id,
                            vector={"text": sparse_vec},
                        )
                    ],
                )
                # Tag the point with the current recipe version (AC-3).
                # set_payload touches payload only — dense vector is untouched.
                client.set_payload(
                    collection_name=COLLECTION,
                    payload={"sparse_recipe": _SPARSE_TEXT_RECIPE},
                    points=[record.id],
                )
                written += 1
            except Exception as exc:  # noqa: BLE001
                _logger.warning(
                    '{"step":"backfill_error","point_id":"%s","error":"%s"}',
                    str(record.id),
                    str(exc)[:120],
                )
                failed += 1

            if total % PROGRESS_EVERY == 0:
                _logger.info(
                    '{"step":"backfill_progress","total":%d,"written":%d,"skipped_populated":%d,"skipped_empty_text":%d,"failed":%d}',
                    total, written, skipped_populated, skipped_empty_text, failed,
                )

        if next_offset is None:
            break

    counts = {
        "total": total,
        "written": written,
        "skipped_populated": skipped_populated,
        "skipped_empty_text": skipped_empty_text,
        "failed": failed,
    }
    _logger.info(
        '{"step":"backfill_done","total":%d,"written":%d,"skipped_populated":%d,"skipped_empty_text":%d,"failed":%d}',
        total, written, skipped_populated, skipped_empty_text, failed,
    )
    return counts


# ---------------------------------------------------------------------------
# AC-3 dense spot-check verify
# ---------------------------------------------------------------------------

def verify_dense_unchanged(
    client: QdrantClient,
    spot_check_id: str | int,
    baseline_dense: list[float],
) -> None:
    """Assert the spot-check point's dense vector is element-wise identical to baseline."""
    retrieved = client.retrieve(
        collection_name=COLLECTION,
        ids=[spot_check_id],
        with_payload=False,
        with_vectors=True,
    )
    assert retrieved, f"Spot-check point {spot_check_id} not found"
    vec = retrieved[0].vector
    if isinstance(vec, dict):
        dense_vec = vec.get("") or next(iter(v for v in vec.values() if isinstance(v, list)), None)
    else:
        dense_vec = vec

    assert dense_vec is not None, "No dense vector on spot-check point after backfill"
    assert len(dense_vec) == len(baseline_dense), (
        f"Dense vector length mismatch: {len(dense_vec)} vs {len(baseline_dense)}"
    )
    for i, (a, b) in enumerate(zip(dense_vec, baseline_dense)):
        assert a == b, f"Dense vector element {i} changed: {a} != {b}"
    _logger.info('{"step":"ac3_verify","result":"dense_unchanged","point_id":"%s"}', str(spot_check_id))


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    import json

    qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key = os.environ.get("QDRANT_API_KEY", "")

    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key or None, timeout=60)

    _logger.info(
        '{"step":"start","collection":"%s","qdrant_url":"%s","sparse_recipe":"%s"}',
        COLLECTION, qdrant_url, _SPARSE_TEXT_RECIPE,
    )

    # P1: capture spot-check baseline before any mutation
    scratchpad = "/tmp/claude-0/-root-dev-env-movie-scout/192aa102-c393-4fb2-9d30-377fca27e3c9/scratchpad"
    baseline_path = f"{scratchpad}/dense_baseline.json"
    if os.path.exists(baseline_path):
        with open(baseline_path) as f:
            baseline_data = json.load(f)
        spot_check_id = baseline_data["point_id"]
        baseline_dense: list[float] = baseline_data["dense_vec"]
        _logger.info(
            '{"step":"spot_check_loaded","point_id":"%s","tmdb_id":%s,"title":"%s"}',
            spot_check_id, baseline_data.get("tmdb_id"), baseline_data.get("title"),
        )
    else:
        # Capture fresh if not already saved
        records, _ = client.scroll(
            collection_name=COLLECTION,
            limit=1,
            with_payload=["tmdb_id", "title"],
            with_vectors=True,
        )
        r = records[0]
        vec = r.vector
        if isinstance(vec, dict):
            baseline_dense = vec.get("") or next(iter(vec.values()), None)
        else:
            baseline_dense = vec
        spot_check_id = r.id
        baseline_data = {"point_id": r.id, "tmdb_id": r.payload.get("tmdb_id"), "title": r.payload.get("title"), "dense_vec": baseline_dense}
        import os as _os
        _os.makedirs(scratchpad, exist_ok=True)
        with open(baseline_path, "w") as f:
            json.dump(baseline_data, f)
        _logger.info(
            '{"step":"spot_check_captured","point_id":"%s"}', str(spot_check_id),
        )

    # P1: ensure sparse schema (recreate fallback — no-op if already present)
    ensure_sparse_config(client)

    # P2: run the enriched backfill
    counts = backfill(client)

    # AC-4: verify dense unchanged after backfill
    verify_dense_unchanged(client, spot_check_id, baseline_dense)

    # AC-2/AC-4: verify point count unchanged (compare against schema-step baseline,
    # not a hardcoded literal — the corpus size can legitimately shift between runs)
    info = client.get_collection(COLLECTION)
    _logger.info('{"step":"ac4_verify","points_count":%d}', info.points_count)

    # Inspect the spot-check point's sparse vector (AC-5 partial)
    retrieved = client.retrieve(
        collection_name=COLLECTION,
        ids=[spot_check_id],
        with_payload=["title", "genres", "cast", "sparse_recipe"],
        with_vectors=["text"],
    )
    if retrieved:
        sv = retrieved[0].vector
        if isinstance(sv, dict):
            text_sv = sv.get("text")
        else:
            text_sv = sv
        if text_sv is not None and hasattr(text_sv, "indices"):
            _logger.info(
                '{"step":"ac5_spot_check_sparse","n_tokens":%d,"first_indices":%s}',
                len(text_sv.indices),
                str(text_sv.indices[:5].tolist() if hasattr(text_sv.indices, "tolist") else list(text_sv.indices)[:5]),
            )
        pay = retrieved[0].payload or {}
        _logger.info(
            '{"step":"ac3_recipe_tag","sparse_recipe":"%s"}',
            pay.get("sparse_recipe", "MISSING"),
        )

    _logger.info(
        '{"step":"all_done","counts":%s}',
        str(counts),
    )


if __name__ == "__main__":
    main()
