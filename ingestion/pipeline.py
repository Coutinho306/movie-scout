"""Entry point for the TMDB ingestion pipeline."""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PayloadSchemaType, VectorParams

from ingestion.resources.tmdb_movies import (
    discover_candidate_tmdb_ids,
    load_tmdb_movies,
)
from ingestion.resources.tmdb_reviews import load_tmdb_reviews
from ingestion.scripts.compute_taste import load_letterboxd_csvs, search_tmdb

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(message)s")
_logger = logging.getLogger(__name__)


def get_qdrant_client(url: str, api_key: str) -> QdrantClient:
    return QdrantClient(url=url, api_key=api_key)


def ensure_collections(client: QdrantClient) -> None:
    existing = {c.name for c in client.get_collections().collections}

    if "tmdb_movies" not in existing:
        client.create_collection(
            collection_name="tmdb_movies",
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )
    client.create_payload_index("tmdb_movies", "tmdb_id", PayloadSchemaType.KEYWORD)
    client.create_payload_index("tmdb_movies", "year", PayloadSchemaType.INTEGER)
    client.create_payload_index("tmdb_movies", "genres", PayloadSchemaType.KEYWORD)
    client.create_payload_index("tmdb_movies", "vote_average", PayloadSchemaType.FLOAT)

    if "tmdb_reviews" not in existing:
        client.create_collection(
            collection_name="tmdb_reviews",
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )
    client.create_payload_index("tmdb_reviews", "tmdb_id", PayloadSchemaType.KEYWORD)


def load_watched_tmdb_ids(export_dir: Path, tmdb_api_key: str) -> set[int]:
    pool, _ = load_letterboxd_csvs(export_dir)
    ids: set[int] = set()
    for film in pool:
        result = search_tmdb(film.name, film.year, tmdb_api_key)
        if result:
            ids.add(result.tmdb_id)
        import time; time.sleep(0.25)
    _logger.info('{"step":"watched_ids_resolved","count":%d}', len(ids))
    return ids


def run_pipeline(
    *,
    tmdb_api_key: str,
    openai_api_key: str,
    qdrant_url: str,
    qdrant_api_key: str,
    discovery_pages: int = 5,
) -> None:
    os.environ["OPENAI_API_KEY"] = openai_api_key

    client = get_qdrant_client(qdrant_url, qdrant_api_key)
    ensure_collections(client)

    export_dir = Path("data/letterboxd_export")
    watched_tmdb_ids = load_watched_tmdb_ids(export_dir, tmdb_api_key)

    _logger.info('{"step":"movies_load_start"}')
    movies_loaded = load_tmdb_movies(
        api_key=tmdb_api_key,
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
        watched_tmdb_ids=watched_tmdb_ids,
        discovery_pages=discovery_pages,
    )

    candidate_ids = discover_candidate_tmdb_ids(tmdb_api_key, pages=discovery_pages)
    candidate_ids = [i for i in candidate_ids if i not in watched_tmdb_ids]

    _logger.info('{"step":"reviews_load_start","candidates":%d}', len(candidate_ids))
    reviews_loaded = load_tmdb_reviews(
        api_key=tmdb_api_key,
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
        candidate_tmdb_ids=candidate_ids,
    )
    _logger.info(
        '{"step":"pipeline_complete","movies_loaded":%d,"reviews_loaded":%d}',
        movies_loaded,
        reviews_loaded,
    )


if __name__ == "__main__":
    run_pipeline(
        tmdb_api_key=os.environ["TMDB_API_KEY"],
        openai_api_key=os.environ["OPENAI_API_KEY"],
        qdrant_url=os.environ["QDRANT_URL"],
        qdrant_api_key=os.environ["QDRANT_API_KEY"],
    )
