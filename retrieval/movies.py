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
    Prefetch,
    Range,
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
