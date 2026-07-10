"""Taste-upload service: parse a Letterboxd export → build a TasteProfile.

Zero OpenAI embedding cost: vectors are pulled directly from the Qdrant corpus
by point id (uuid5(NAMESPACE_DNS, str(tmdb_id))) rather than re-embedded.

Pipeline:
  1. parse_letterboxd_pool(bytes, filename) → weighted LetterboxdFilm pool
  2. Cap to top-N by rating_weight before TMDB lookups
  3. search_tmdb per film (skip+count misses)
  4. batch client.retrieve by derived point id (with_vectors=True, with_payload=True)
  5. compute_centroid + rank_genre_ids → TasteProfile
  6. Return (TasteProfile, ResolutionReport)
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from ingestion.models import LetterboxdFilm, TasteProfile
from ingestion.scripts.compute_taste import (
    compute_centroid,
    parse_letterboxd_pool,
    rank_genre_ids,
    rating_weight,
    search_tmdb,
)
from retrieval.client import get_qdrant_client

logger = logging.getLogger(__name__)

_COLLECTION = "tmdb_movies"
_DEFAULT_TOP_N_FILMS = 500  # cap before TMDB resolution; balances fidelity vs latency


def _point_id(tmdb_id: int) -> str:
    """Derive the Qdrant point id from a TMDB movie id (mirrors ingestion)."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, str(tmdb_id)))


@dataclass
class ResolutionReport:
    resolved: int = 0
    tmdb_miss: int = 0
    out_of_corpus: int = 0
    skipped_zero_weight: int = 0
    total_input: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "resolved": self.resolved,
            "tmdb_miss": self.tmdb_miss,
            "out_of_corpus": self.out_of_corpus,
            "skipped_zero_weight": self.skipped_zero_weight,
            "total_input": self.total_input,
        }


@dataclass
class TasteUploadResult:
    profile: TasteProfile
    report: ResolutionReport


def build_taste_profile_from_upload(
    data: bytes,
    *,
    filename: str = "",
    tmdb_api_key: str,
    top_n_films: int = _DEFAULT_TOP_N_FILMS,
    top_n_genres: int = 4,
    tmdb_sleep: float = 0.25,
    qdrant_url: str | None = None,
    qdrant_api_key: str | None = None,
) -> TasteUploadResult:
    """Build a TasteProfile from a raw Letterboxd export (CSV or ZIP bytes).

    Args:
        data: Raw bytes of the uploaded file.
        filename: Original filename; used to detect ZIP vs CSV.
        tmdb_api_key: TMDB bearer token for title resolution.
        top_n_films: Cap — only the top-N films by rating_weight are resolved
            against TMDB. Applied before any HTTP calls. Default: 500.
        top_n_genres: Number of top genre IDs in the profile. Default: 4.
        tmdb_sleep: Seconds to sleep between TMDB requests. Default: 0.25.
        qdrant_url: Override Qdrant URL (defaults to QDRANT_URL env var).
        qdrant_api_key: Override Qdrant API key.

    Returns:
        TasteUploadResult with profile + resolution report.

    Raises:
        ValueError: If the upload is not a valid Letterboxd CSV/ZIP.
        RuntimeError: If fewer than 2 in-corpus films are found (cannot form centroid).
    """
    report = ResolutionReport()

    # --- Step 1: Parse ---
    pool, _ = parse_letterboxd_pool(data, filename=filename)
    report.total_input = len(pool)

    # --- Step 2: Weight + cap to top-N ---
    weighted_pool: list[tuple[LetterboxdFilm, float]] = []
    for film in pool:
        w = rating_weight(film)
        if w <= 0.0:
            report.skipped_zero_weight += 1
            continue
        weighted_pool.append((film, w))

    # Sort descending by weight, cap to top_n_films
    weighted_pool.sort(key=lambda fw: fw[1], reverse=True)
    if len(weighted_pool) > top_n_films:
        weighted_pool = weighted_pool[:top_n_films]

    if not weighted_pool:
        raise ValueError(
            "No films with positive weight found in the upload. "
            "Ensure ratings.csv has rated films (rating ≥ 1.5)."
        )

    # --- Step 3: Resolve titles via TMDB (skip+count misses) ---
    resolved: list[tuple[int, float]] = []  # (tmdb_id, weight)

    for film, weight in weighted_pool:
        logger.debug(
            '{"step":"search_tmdb","film":"%s","year":%d}', film.name, film.year
        )
        result = search_tmdb(film.name, film.year, tmdb_api_key)
        if tmdb_sleep > 0:
            time.sleep(tmdb_sleep)
        if result is None:
            logger.info(
                '{"step":"tmdb_miss","film":"%s","year":%d}', film.name, film.year
            )
            report.tmdb_miss += 1
            continue
        resolved.append((result.tmdb_id, weight))

    if not resolved:
        raise ValueError(
            "No films resolved via TMDB. All title lookups failed."
        )

    # --- Step 4: Batch retrieve vectors from Qdrant by point id ---
    client = get_qdrant_client(url=qdrant_url, api_key=qdrant_api_key)

    # Build id → weight map (dedup: keep highest weight per tmdb_id)
    id_to_weight: dict[int, float] = {}
    for tmdb_id, weight in resolved:
        if tmdb_id not in id_to_weight or weight > id_to_weight[tmdb_id]:
            id_to_weight[tmdb_id] = weight

    point_ids = [_point_id(tid) for tid in id_to_weight]
    pid_to_tmdb = {_point_id(tid): tid for tid in id_to_weight}

    records = client.retrieve(
        collection_name=_COLLECTION,
        ids=point_ids,
        with_vectors=True,
        with_payload=True,
    )

    in_corpus_ids = {str(r.id) for r in records}
    out_of_corpus = len(point_ids) - len(records)
    report.out_of_corpus = out_of_corpus
    report.resolved = len(records)

    logger.info(
        '{"step":"batch_retrieve","requested":%d,"found":%d,"out_of_corpus":%d}',
        len(point_ids),
        len(records),
        out_of_corpus,
    )

    if not records:
        raise RuntimeError(
            "No in-corpus films found. Cannot compute taste centroid without "
            "at least one film present in the tmdb_movies collection."
        )

    # --- Step 5: Assemble centroid + genres ---
    vectors: list[list[float]] = []
    weights: list[float] = []
    genre_weights: dict[str, float] = {}
    titled_weights: list[tuple[float, str]] = []
    rated_count = 0
    liked_count = 0

    for record in records:
        pid = str(record.id)
        tmdb_id = pid_to_tmdb[pid]
        weight = id_to_weight[tmdb_id]

        # Unwrap dict vector (named-vector collection: key '' is the dense vec)
        vec = record.vector
        if isinstance(vec, dict):
            vec = vec.get("") or next(iter(vec.values()), None)
        if not vec:
            logger.warning(
                '{"step":"skip_no_vector","tmdb_id":%d}', tmdb_id
            )
            continue

        vectors.append(list(vec))
        weights.append(weight)

        # Genres + title from corpus payload (avoids a second TMDB fetch)
        payload_genres: list[str] = []
        title = ""
        if record.payload:
            raw = record.payload.get("genres", [])
            if isinstance(raw, list):
                payload_genres = [str(g) for g in raw]
            title = str(record.payload.get("title", ""))
        for genre in payload_genres:
            genre_weights[genre] = genre_weights.get(genre, 0.0) + weight
        if title:
            titled_weights.append((weight, title))

        # Approximation: weight=1.0 → rated high; weight=0.7 → liked
        if weight >= 0.8:
            rated_count += 1
        elif weight >= 0.65:
            liked_count += 1

    if len(vectors) < 1:
        raise RuntimeError(
            "No valid vectors extracted from corpus. Cannot compute centroid."
        )

    centroid = compute_centroid(vectors, weights)
    top_genre_ids = rank_genre_ids(genre_weights, top_n=top_n_genres)
    top_films = [
        title for _, title in sorted(titled_weights, key=lambda t: t[0], reverse=True)[:5]
    ]

    profile = TasteProfile(
        centroid=centroid,
        film_count=len(vectors),
        rated_count=rated_count,
        liked_count=liked_count,
        top_genre_ids=top_genre_ids,
        genre_weights=genre_weights,
        created_at=datetime.now(timezone.utc).isoformat(),
        top_films=top_films,
    )

    logger.info(
        '{"step":"taste_upload_done","film_count":%d,"tmdb_miss":%d,"out_of_corpus":%d}',
        profile.film_count,
        report.tmdb_miss,
        report.out_of_corpus,
    )

    return TasteUploadResult(profile=profile, report=report)
