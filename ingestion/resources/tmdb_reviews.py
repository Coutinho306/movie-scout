"""TMDB reviews — fetch, chunk, embed, load Qdrant."""

import logging
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from ingestion.chunking import chunk_review
from ingestion.embedding import Embedder
from ingestion.resources.tmdb_movies import TMDB_BASE, tmdb_get

_logger = logging.getLogger(__name__)


def fetch_reviews(tmdb_id: int, api_key: str) -> list[dict]:
    resp = tmdb_get(
        f"{TMDB_BASE}/movie/{tmdb_id}/reviews",
        api_key=api_key,
        params={"page": 1},
    )
    if resp.status_code != 200:
        return []
    return resp.json().get("results", [])


def load_tmdb_reviews(
    api_key: str,
    qdrant_url: str,
    qdrant_api_key: str,
    candidate_tmdb_ids: list[int],
    embedder: Embedder,
    collection_name: str,
    *,
    chunk_max_tokens: int = 300,
    chunk_overlap_tokens: int = 50,
) -> int:
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=30)
    loaded = 0

    for tmdb_id in candidate_tmdb_ids:
        reviews = fetch_reviews(tmdb_id, api_key)

        for review in reviews:
            author = review.get("author", "unknown")
            content = review.get("content", "")
            if not content.strip():
                continue

            chunks = chunk_review(
                content,
                max_tokens=chunk_max_tokens,
                overlap_tokens=chunk_overlap_tokens,
            )
            vectors = embedder.embed_texts(chunks)

            for chunk_index, (chunk_text, vector) in enumerate(zip(chunks, vectors)):
                point_id = str(
                    uuid.uuid5(uuid.NAMESPACE_DNS, f"{tmdb_id}_{author}_{chunk_index}")
                )
                client.upsert(
                    collection_name=collection_name,
                    points=[
                        PointStruct(
                            id=point_id,
                            vector=vector,
                            payload={
                                "tmdb_id": tmdb_id,
                                "review_author": author,
                                "chunk_index": chunk_index,
                                "total_chunks": len(chunks),
                                "chunk_text": chunk_text,
                            },
                        )
                    ],
                )
                loaded += 1

    _logger.info('{"step":"tmdb_reviews_done","loaded":%d}', loaded)
    return loaded
