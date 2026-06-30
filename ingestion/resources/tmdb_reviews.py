"""TMDB reviews — fetch, chunk, embed, load Qdrant."""

import logging
import time
import uuid

import requests
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from ingestion.chunking import chunk_review
from ingestion.embedding import Embedder

_logger = logging.getLogger(__name__)
TMDB_BASE = "https://api.themoviedb.org/3"


def fetch_reviews(tmdb_id: int, api_key: str) -> list[dict]:
    resp = requests.get(
        f"{TMDB_BASE}/movie/{tmdb_id}/reviews",
        params={"page": 1},
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=10,
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
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    loaded = 0

    for tmdb_id in candidate_tmdb_ids:
        reviews = fetch_reviews(tmdb_id, api_key)
        time.sleep(0.25)

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
