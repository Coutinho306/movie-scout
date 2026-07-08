"""Search tmdb_reviews collection (dense vector only).

``tmdb_reviews`` has no sparse ``text`` field — hybrid RRF is not supported
here. ``search_reviews`` is unconditionally dense-only. If reviews hybrid is
ever wanted it requires a separate sparse backfill (see specs/0009).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from qdrant_client.models import FieldCondition, Filter, MatchAny, ScoredPoint

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
    """Search tmdb_reviews collection with dense vector retrieval.

    Pass tmdb_ids to restrict results to specific films.
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

    results = client.query_points(
        collection_name=collection,
        query=query_vec,
        limit=limit,
        query_filter=qdrant_filter,
        score_threshold=settings.score_threshold,
        with_payload=True,
    ).points

    return [_point_to_hit(p) for p in results]
