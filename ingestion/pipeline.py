"""Entry point for the TMDB ingestion pipeline."""

import argparse
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PayloadSchemaType, VectorParams

from ingestion.models import TasteProfile
from ingestion.resources.tmdb_movies import (
    DISCOVERY_GENRES,
    discover_candidate_tmdb_ids,
    load_tmdb_movies,
)
from ingestion.resources.tmdb_reviews import load_tmdb_reviews
from ingestion.scripts.compute_taste import (
    compute_taste_profile,
    load_letterboxd_csvs,
    search_tmdb,
)

_TASTE_PROFILE_PATH = Path("data/taste_profile.json")

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


def rebuild_collections(client: QdrantClient) -> None:
    """Drop both collections so ensure_collections recreates them empty."""
    existing = {c.name for c in client.get_collections().collections}
    for name in ("tmdb_movies", "tmdb_reviews"):
        if name in existing:
            client.delete_collection(name)
            _logger.info('{"step":"collection_dropped","collection":"%s"}', name)


def load_or_compute_taste(
    tmdb_api_key: str, *, refresh: bool, skip: bool
) -> TasteProfile:
    """Ensure a taste profile exists, then return it.

    compute_taste is a hard dependency: the profile's top_genre_ids drive
    discovery. Recompute when missing or when refresh is requested; skip forces
    reuse of an existing profile and errors if none is present.
    """
    if skip:
        if not _TASTE_PROFILE_PATH.exists():
            raise FileNotFoundError(
                f"--skip-taste set but {_TASTE_PROFILE_PATH} does not exist"
            )
    elif refresh or not _TASTE_PROFILE_PATH.exists():
        _logger.info('{"step":"taste_compute_start"}')
        return compute_taste_profile(tmdb_api_key, output_path=_TASTE_PROFILE_PATH)

    return TasteProfile.model_validate_json(_TASTE_PROFILE_PATH.read_text())


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
    rebuild: bool = False,
    refresh_taste: bool = False,
    skip_taste: bool = False,
) -> None:
    os.environ["OPENAI_API_KEY"] = openai_api_key

    taste = load_or_compute_taste(
        tmdb_api_key, refresh=refresh_taste, skip=skip_taste
    )
    genre_ids = taste.top_genre_ids or DISCOVERY_GENRES
    _logger.info('{"step":"discovery_genres","genre_ids":%s}', json.dumps(genre_ids))

    client = get_qdrant_client(qdrant_url, qdrant_api_key)
    if rebuild:
        rebuild_collections(client)
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
        genre_ids=genre_ids,
    )

    candidate_ids = discover_candidate_tmdb_ids(
        tmdb_api_key, pages=discovery_pages, genre_ids=genre_ids
    )
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TMDB ingestion pipeline")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="drop and recreate Qdrant collections before loading",
    )
    parser.add_argument(
        "--refresh-taste",
        action="store_true",
        help="recompute taste_profile.json even if it already exists",
    )
    parser.add_argument(
        "--skip-taste",
        action="store_true",
        help="reuse the existing taste_profile.json without recomputing",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_pipeline(
        tmdb_api_key=os.environ["TMDB_API_KEY"],
        openai_api_key=os.environ["OPENAI_API_KEY"],
        qdrant_url=os.environ["QDRANT_URL"],
        qdrant_api_key=os.environ["QDRANT_API_KEY"],
        rebuild=args.rebuild,
        refresh_taste=args.refresh_taste,
        skip_taste=args.skip_taste,
    )
