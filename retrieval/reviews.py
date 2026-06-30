"""Search tmdb_reviews collection."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from qdrant_client.models import FieldCondition, Filter, Fusion, MatchAny, Prefetch, ScoredPoint

from ingestion.embedding import get_embedder
from retrieval.client import get_qdrant_client
from retrieval.models import ReviewHit

if TYPE_CHECKING:
    from retrieval.config import RetrievalSettings

_logger = logging.getLogger(__name__)


def _point_to_hit(point: ScoredPoint) -> ReviewHit:
    p = point.payload or {}
    return ReviewHit(
        tmdb_id=p.get("tmdb_id", 0),
        title=p.get("title", ""),
        review_author=p.get("review_author", ""),
        chunk_text=p.get("chunk_text", ""),
        chunk_index=p.get("chunk_index", 0),
        score=point.score,
    )


def search_reviews(
    query: str,
    *,
    settings: RetrievalSettings,
    k: int | None = None,
    tmdb_ids: list[int] | None = None,
) -> list[ReviewHit]:
    """Search tmdb_reviews collection.

    Pass tmdb_ids to restrict results to specific films.
    When settings.hybrid=True attempts native RRF; falls back to vector on error.
    """
    ingestion = settings.ingestion()
    collection = ingestion.reviews_collection
    limit = k if k is not None else settings.top_k
    embedder = get_embedder(ingestion)
    query_vec = embedder.embed_single(query)
    client = get_qdrant_client()

    qdrant_filter: Filter | None = None
    if tmdb_ids:
        qdrant_filter = Filter(
            must=[
                FieldCondition(
                    key="tmdb_id",
                    match=MatchAny(any=[str(i) for i in tmdb_ids]),
                )
            ]
        )

    if settings.hybrid:
        try:
            results = client.query_points(
                collection_name=collection,
                prefetch=[
                    Prefetch(query=query_vec, using="", limit=limit * 2),
                    Prefetch(query=query, using="text", limit=limit * 2),
                ],
                query=Fusion.RRF,
                limit=limit,
                query_filter=qdrant_filter,
                score_threshold=settings.score_threshold,
                with_payload=True,
            ).points
        except Exception as exc:
            _logger.warning(
                '{"step":"reviews_hybrid_fallback","reason":"%s"}', str(exc)[:120]
            )
            results = client.query_points(
                collection_name=collection,
                query=query_vec,
                limit=limit,
                query_filter=qdrant_filter,
                score_threshold=settings.score_threshold,
                with_payload=True,
            ).points
    else:
        results = client.query_points(
            collection_name=collection,
            query=query_vec,
            limit=limit,
            query_filter=qdrant_filter,
            score_threshold=settings.score_threshold,
            with_payload=True,
        ).points

    return [_point_to_hit(p) for p in results]
