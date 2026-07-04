"""Search tmdb_movies collection: vector, hybrid (BM25+dense), filters."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from qdrant_client.models import (
    FieldCondition,
    Filter,
    Fusion,
    MatchAny,
    MatchValue,
    NearestQuery,
    Prefetch,
    Range,
    Record,
    ScoredPoint,
)

from ingestion.embedding import get_embedder
from retrieval.client import get_qdrant_client
from retrieval.models import MovieFilters, MovieHit

if TYPE_CHECKING:
    from retrieval.config import RetrievalSettings

_logger = logging.getLogger(__name__)


def _build_filter(filters: MovieFilters | None) -> Filter | None:
    """Build Qdrant Filter from MovieFilters.

    exclude_tmdb_ids is handled in Python post-fetch (tmdb_id KEYWORD index
    doesn't support range/except on integer payload values in Qdrant Cloud).
    """
    if filters is None:
        return None

    conditions = []

    if filters.year_min is not None or filters.year_max is not None:
        conditions.append(
            FieldCondition(
                key="year",
                range=Range(
                    gte=filters.year_min,
                    lte=filters.year_max,
                ),
            )
        )

    if filters.genres:
        conditions.append(
            FieldCondition(
                key="genres",
                match=MatchAny(any=filters.genres),
            )
        )

    if filters.cast:
        conditions.append(
            FieldCondition(
                key="cast",
                match=MatchAny(any=filters.cast),
            )
        )

    if filters.vote_min is not None:
        conditions.append(
            FieldCondition(
                key="vote_average",
                range=Range(gte=filters.vote_min),
            )
        )

    return Filter(must=conditions) if conditions else None


def _extract_vector(point: ScoredPoint) -> list[float] | None:
    """Pull the default dense vector off a point.

    Qdrant returns the vector as a bare list for a single unnamed vector, or as a
    ``{name: vector}`` dict when named/hybrid vectors are configured.
    """
    vec = point.vector
    if isinstance(vec, dict):
        vec = vec.get("") or next(iter(vec.values()), None)
    return vec  # type: ignore[return-value]


def _point_to_hit(point: ScoredPoint) -> MovieHit:
    p = point.payload or {}
    return MovieHit(
        tmdb_id=p.get("tmdb_id", 0),
        title=p.get("title", ""),
        year=p.get("year", 0),
        overview=p.get("overview", ""),
        genres=p.get("genres", []),
        vote_average=p.get("vote_average", 0.0),
        score=point.score,
        vector=_extract_vector(point),
    )


def _record_to_hit(record: Record) -> MovieHit:
    """Map a scroll Record (no .score field) to MovieHit with score=0.0."""
    p = record.payload or {}
    return MovieHit(
        tmdb_id=p.get("tmdb_id", 0),
        title=p.get("title", ""),
        year=p.get("year", 0),
        overview=p.get("overview", ""),
        genres=p.get("genres", []),
        vote_average=p.get("vote_average", 0.0),
        score=0.0,
    )


def list_movies_by_cast(
    actor: str,
    *,
    settings: RetrievalSettings,
    k: int | None = None,
) -> list[MovieHit]:
    """Return all movies in the corpus that list ``actor`` in their cast payload.

    Uses Qdrant scroll with an exact cast MatchAny filter — exhaustive listing,
    no query vector, no semantic search. Results are unranked (score=0.0).

    ``k`` caps the returned list; when None, settings.top_k is used.
    """
    collection = settings.ingestion().movies_collection
    limit = k if k is not None else settings.top_k
    client = get_qdrant_client()

    cast_filter = Filter(
        must=[
            FieldCondition(
                key="cast",
                match=MatchAny(any=[actor]),
            )
        ]
    )

    hits: list[MovieHit] = []
    next_offset = None
    page_size = 100  # scroll page size

    while len(hits) < limit:
        batch_limit = min(page_size, limit - len(hits))
        records, next_offset = client.scroll(
            collection_name=collection,
            scroll_filter=cast_filter,
            limit=batch_limit,
            offset=next_offset,
            with_payload=True,
            with_vectors=False,
        )
        hits.extend(_record_to_hit(r) for r in records)
        if next_offset is None:
            break  # no more pages

    return hits[:limit]


def find_by_exact_title(
    title: str,
    *,
    settings: RetrievalSettings,
) -> list[MovieHit]:
    """Return every film in the corpus whose title exactly matches ``title``.

    Uses Qdrant scroll with FieldCondition(key="title", MatchValue) — no
    embedding, no vector search. Reliably surfaces the full collision set for
    same-title disambiguation (e.g. four films all titled "Obsession").

    Returns [] if no match; returns the full match set (all pages) if multiple
    films share the title.
    """
    collection = settings.ingestion().movies_collection
    client = get_qdrant_client()

    title_filter = Filter(
        must=[
            FieldCondition(
                key="title",
                match=MatchValue(value=title),
            )
        ]
    )

    hits: list[MovieHit] = []
    next_offset = None

    while True:
        records, next_offset = client.scroll(
            collection_name=collection,
            scroll_filter=title_filter,
            limit=100,
            offset=next_offset,
            with_payload=True,
            with_vectors=False,
        )
        hits.extend(_record_to_hit(r) for r in records)
        if next_offset is None:
            break

    return hits


def search_movies(
    query: str,
    *,
    settings: RetrievalSettings,
    k: int | None = None,
    filters: MovieFilters | None = None,
) -> list[MovieHit]:
    """Search tmdb_movies with vector (or hybrid) retrieval.

    When settings.hybrid=True tries Qdrant native RRF fusion. If the collection
    lacks a sparse vector field the call falls back to dense-only with a warning.
    """
    ingestion = settings.ingestion()
    collection = ingestion.movies_collection
    limit = k if k is not None else settings.top_k
    embedder = get_embedder(ingestion)

    if settings.query_rewrite:
        from retrieval.hyde import hyde_embed

        _blend_raw = os.environ.get("HYDE_BLEND_ALPHA", "").strip()
        blend_alpha: float | None = float(_blend_raw) if _blend_raw else None
        query_vec = hyde_embed(query, embedder, blend_alpha=blend_alpha)
        _logger.debug(
            '{"step":"hyde","blend_alpha":%s}',
            repr(blend_alpha),
        )
    else:
        query_vec = embedder.embed_single(query)

    client = get_qdrant_client()
    qdrant_filter = _build_filter(filters)

    if settings.hybrid:
        try:
            results = client.query_points(
                collection_name=collection,
                prefetch=[
                    Prefetch(
                        query=query_vec,
                        using="",  # default dense vector
                        limit=limit * 2,
                    ),
                    Prefetch(
                        query=query,  # sparse text query
                        using="text",  # sparse vector field name
                        limit=limit * 2,
                    ),
                ],
                query=Fusion.RRF,
                limit=limit,
                query_filter=qdrant_filter,
                score_threshold=settings.score_threshold,
                with_payload=True,
                with_vectors=True,
            ).points
        except Exception as exc:
            _logger.warning(
                '{"step":"hybrid_fallback","reason":"%s","collection":"%s"}',
                str(exc)[:120],
                collection,
            )
            results = client.query_points(
                collection_name=collection,
                query=query_vec,
                limit=limit,
                query_filter=qdrant_filter,
                score_threshold=settings.score_threshold,
                with_payload=True,
                with_vectors=True,
            ).points
    else:
        results = client.query_points(
            collection_name=collection,
            query=query_vec,
            limit=limit,
            query_filter=qdrant_filter,
            score_threshold=settings.score_threshold,
            with_payload=True,
            with_vectors=True,
        ).points

    hits = [_point_to_hit(p) for p in results]

    # Post-fetch exclusion: Qdrant KEYWORD index on integer tmdb_id doesn't
    # support MatchExcept reliably, so we filter in Python.
    if filters and filters.exclude_tmdb_ids:
        hits = [h for h in hits if h.tmdb_id not in filters.exclude_tmdb_ids]

    return hits


def recommend_similar(
    seed_tmdb_id: int,
    *,
    settings: RetrievalSettings,
    k: int | None = None,
    filters: MovieFilters | None = None,
) -> list[MovieHit]:
    """Return movies similar to the seed film using its stored dense vector.

    Steps:
    1. Resolve the seed's Qdrant point id by scrolling for payload tmdb_id.
    2. Retrieve that point with_vectors=True to get its stored dense vector.
    3. Run a point-to-point vector similarity query (NearestQuery) against
       the collection, applying _build_filter + post-fetch exclude_tmdb_ids.

    The seed's own tmdb_id is always added to exclude_tmdb_ids so it can never
    appear in the results (self-recommendation fix).

    Returns [] if the seed point is not found in the corpus so the caller can
    fall back to text search.
    """
    collection = settings.ingestion().movies_collection
    limit = k if k is not None else settings.top_k
    client = get_qdrant_client()

    # Step 1: find the seed's point id by filtering payload tmdb_id.
    seed_filter = Filter(
        must=[
            FieldCondition(
                key="tmdb_id",
                match=MatchValue(value=seed_tmdb_id),
            )
        ]
    )
    records, _ = client.scroll(
        collection_name=collection,
        scroll_filter=seed_filter,
        limit=1,
        with_payload=False,
        with_vectors=False,
    )
    if not records:
        _logger.debug(
            '{"step":"recommend_similar","seed_tmdb_id":%d,"found":false}',
            seed_tmdb_id,
        )
        return []

    seed_point_id = records[0].id

    # Step 2: retrieve the seed point with its stored dense vector.
    retrieved = client.retrieve(
        collection_name=collection,
        ids=[seed_point_id],
        with_payload=False,
        with_vectors=True,
    )
    if not retrieved:
        return []

    seed_record = retrieved[0]
    vec = seed_record.vector
    if isinstance(vec, dict):
        vec = vec.get("") or next(iter(vec.values()), None)
    if not vec:
        _logger.warning(
            '{"step":"recommend_similar","seed_tmdb_id":%d,"error":"no_vector"}',
            seed_tmdb_id,
        )
        return []

    # Step 3: build exclusion set — seed always excluded.
    exclude_ids: set[int] = {seed_tmdb_id}
    if filters and filters.exclude_tmdb_ids:
        exclude_ids |= filters.exclude_tmdb_ids

    effective_filters = MovieFilters(
        year_min=filters.year_min if filters else None,
        year_max=filters.year_max if filters else None,
        genres=filters.genres if filters else None,
        vote_min=filters.vote_min if filters else None,
        exclude_tmdb_ids=None,  # handled post-fetch
    )
    qdrant_filter = _build_filter(effective_filters)

    results = client.query_points(
        collection_name=collection,
        query=NearestQuery(nearest=list(vec)),
        limit=limit,
        query_filter=qdrant_filter,
        with_payload=True,
        with_vectors=True,
    ).points

    hits = [_point_to_hit(p) for p in results]

    # Post-fetch exclusion (same pattern as search_movies).
    hits = [h for h in hits if h.tmdb_id not in exclude_ids]

    return hits
