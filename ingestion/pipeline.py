"""Entry point for the TMDB ingestion pipeline."""

import argparse
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Modifier,
    PayloadSchemaType,
    SparseVectorParams,
    VectorParams,
)

from ingestion.config import Settings
from ingestion.embedding import Embedder, get_embedder
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
    return QdrantClient(url=url, api_key=api_key, timeout=30)


def ensure_collections(client: QdrantClient, settings: Settings) -> None:
    existing = {c.name for c in client.get_collections().collections}
    dim = settings.embedder_dim
    movies_col = settings.movies_collection
    reviews_col = settings.reviews_collection

    if movies_col not in existing:
        # tmdb_movies always gets a BM25 sparse "text" vector alongside dense so
        # hybrid (RRF) search works regardless of ingest variant. A prior version
        # of this gated sparse config on settings.sample, which meant any full
        # rebuild of the production collection (--rebuild, or a manual
        # delete+recreate) silently dropped hybrid search — see
        # scripts/backfill_bm25_sparse.py for the one-off repair this required.
        client.create_collection(
            collection_name=movies_col,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            sparse_vectors_config={"text": SparseVectorParams(modifier=Modifier.IDF)},
        )
    client.create_payload_index(movies_col, "tmdb_id", PayloadSchemaType.KEYWORD)
    client.create_payload_index(movies_col, "year", PayloadSchemaType.INTEGER)
    client.create_payload_index(movies_col, "genres", PayloadSchemaType.KEYWORD)
    client.create_payload_index(movies_col, "vote_average", PayloadSchemaType.FLOAT)
    client.create_payload_index(movies_col, "cast", PayloadSchemaType.KEYWORD)
    client.create_payload_index(movies_col, "title", PayloadSchemaType.KEYWORD)
    client.create_payload_index(movies_col, "keywords", PayloadSchemaType.KEYWORD)

    if reviews_col not in existing:
        client.create_collection(
            collection_name=reviews_col,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
    client.create_payload_index(reviews_col, "tmdb_id", PayloadSchemaType.KEYWORD)


def rebuild_collections(client: QdrantClient, settings: Settings) -> None:
    """Drop this variant's two collections so ensure_collections recreates them empty."""
    existing = {c.name for c in client.get_collections().collections}
    for name in (settings.movies_collection, settings.reviews_collection):
        if name in existing:
            client.delete_collection(name)
            _logger.info('{"step":"collection_dropped","collection":"%s"}', name)


def drop_variant(client: QdrantClient, settings: Settings) -> None:
    """Delete this variant's collections without recreating."""
    rebuild_collections(client, settings)


def load_or_compute_taste(
    tmdb_api_key: str, *, refresh: bool, skip: bool, embedder: Embedder
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
        return compute_taste_profile(
            tmdb_api_key, embedder=embedder, output_path=_TASTE_PROFILE_PATH
        )

    return TasteProfile.model_validate_json(_TASTE_PROFILE_PATH.read_text())


def load_watched_tmdb_ids(export_dir: Path, tmdb_api_key: str) -> set[int]:
    from ingestion.resources.tmdb_movies import FETCH_PACE_SECONDS
    import time

    pool, _ = load_letterboxd_csvs(export_dir)
    ids: set[int] = set()
    for film in pool:
        result = search_tmdb(film.name, film.year, tmdb_api_key)
        if result:
            ids.add(result.tmdb_id)
        time.sleep(FETCH_PACE_SECONDS)
    _logger.info('{"step":"watched_ids_resolved","count":%d}', len(ids))
    return ids


def run_pipeline(
    *,
    tmdb_api_key: str,
    openai_api_key: str,
    qdrant_url: str,
    qdrant_api_key: str,
    settings: Settings,
    discovery_pages: int = 5,
    rebuild: bool = False,
    refresh_taste: bool = False,
    skip_taste: bool = False,
    explicit_tmdb_ids: list[int] | None = None,
    resume: bool = False,
    workers: int = 8,
    skip_reviews: bool = False,
) -> None:
    os.environ["OPENAI_API_KEY"] = openai_api_key

    embedder = get_embedder(settings)
    _logger.info(
        '{"step":"variant","embedder":"%s","dim":%d,"movies_collection":"%s"}',
        settings.embedder,
        settings.embedder_dim,
        settings.movies_collection,
    )

    client = get_qdrant_client(qdrant_url, qdrant_api_key)

    if rebuild:
        rebuild_collections(client, settings)
    ensure_collections(client, settings)

    if explicit_tmdb_ids is not None:
        # Calibration sample path: a fixed id list, no taste/discovery. The same
        # ids feed movies and reviews so every variant indexes an identical corpus.
        _logger.info(
            '{"step":"sample_mode","ids":%d,"recipe":"%s"}',
            len(explicit_tmdb_ids),
            settings.embed_text_recipe,
        )
        movies_loaded = load_tmdb_movies(
            api_key=tmdb_api_key,
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key,
            watched_tmdb_ids=set(),
            embedder=embedder,
            collection_name=settings.movies_collection,
            explicit_tmdb_ids=explicit_tmdb_ids,
            embed_text_recipe=settings.embed_text_recipe,
            sparse=True,  # see ensure_collections call below for why this is always True
            resume=resume,
            workers=workers,
        )
        candidate_ids = list(explicit_tmdb_ids)
        if skip_reviews:
            reviews_loaded = 0
            _logger.info('{"step":"reviews_load_skipped","reason":"skip_reviews"}')
        else:
            _logger.info('{"step":"reviews_load_start","candidates":%d}', len(candidate_ids))
            reviews_loaded = load_tmdb_reviews(
                api_key=tmdb_api_key,
                qdrant_url=qdrant_url,
                qdrant_api_key=qdrant_api_key,
                candidate_tmdb_ids=candidate_ids,
                embedder=embedder,
                collection_name=settings.reviews_collection,
                chunk_max_tokens=settings.chunk_max_tokens,
                chunk_overlap_tokens=settings.chunk_overlap_tokens,
                resume=resume,
            )
        _logger.info(
            '{"step":"pipeline_complete","movies_loaded":%d,"reviews_loaded":%d}',
            movies_loaded,
            reviews_loaded,
        )
        return

    taste = load_or_compute_taste(
        tmdb_api_key, refresh=refresh_taste, skip=skip_taste, embedder=embedder
    )
    genre_ids = taste.top_genre_ids or DISCOVERY_GENRES
    _logger.info('{"step":"discovery_genres","genre_ids":%s}', json.dumps(genre_ids))

    export_dir = Path("data/letterboxd_export")
    watched_tmdb_ids = load_watched_tmdb_ids(export_dir, tmdb_api_key)

    _logger.info('{"step":"movies_load_start"}')
    movies_loaded = load_tmdb_movies(
        api_key=tmdb_api_key,
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
        watched_tmdb_ids=watched_tmdb_ids,
        embedder=embedder,
        collection_name=settings.movies_collection,
        discovery_pages=discovery_pages,
        genre_ids=genre_ids,
        embed_text_recipe=settings.embed_text_recipe,
        # tmdb_movies always has a sparse "text" vector field (ensure_collections
        # above), so every point must write one too — omitting this here was the
        # original bug: the schema existed but production points stayed dense-only.
        sparse=True,
        resume=resume,
        workers=workers,
    )

    if skip_reviews:
        reviews_loaded = 0
        _logger.info('{"step":"reviews_load_skipped","reason":"skip_reviews"}')
    else:
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
            embedder=embedder,
            collection_name=settings.reviews_collection,
            chunk_max_tokens=settings.chunk_max_tokens,
            chunk_overlap_tokens=settings.chunk_overlap_tokens,
            resume=resume,
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
        settings=Settings(),
        rebuild=args.rebuild,
        refresh_taste=args.refresh_taste,
        skip_taste=args.skip_taste,
    )
