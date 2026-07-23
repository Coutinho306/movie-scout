"""TMDB candidate pool — fetch metadata, embed, load Qdrant."""

import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import requests
from qdrant_client import QdrantClient, models
from qdrant_client.models import PointStruct

from ingestion.chunking import build_movie_embed_text, build_sparse_text
from ingestion.embedding import Embedder
from ingestion.models import TmdbMovieMetadata

# Page size for scroll-based existing-id enumeration.
_SCROLL_PAGE_SIZE: int = 100

_logger = logging.getLogger(__name__)
TMDB_BASE = "https://api.themoviedb.org/3"
DISCOVERY_GENRES = [18, 28, 53]  # Drama, Action, Thriller

# Default inter-request delay in seconds.  Well under TMDB's ~50 req/s limit;
# keep ≤ 0.05 so ~15 k films + reviews complete under 1 h wall clock.
FETCH_PACE_SECONDS: float = 0.05

# Retry budget for transient TMDB errors (429 / 5xx).
_MAX_RETRIES: int = 5
_RETRY_BACKOFF_BASE: float = 2.0  # seconds; doubles each attempt


def tmdb_get(
    url: str,
    *,
    api_key: str,
    params: dict | None = None,
    timeout: int = 20,
) -> requests.Response:
    """GET a TMDB URL with automatic retry on 429/5xx and read timeouts.

    Respects the ``Retry-After`` response header when present.
    Raises ``requests.HTTPError`` for non-retryable 4xx errors and for
    exhausted retries.  Applies ``FETCH_PACE_SECONDS`` after each successful
    (non-retry) request.
    """
    headers = {"Authorization": f"Bearer {api_key}"}
    attempt = 0
    while True:
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=timeout)
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            attempt += 1
            if attempt > _MAX_RETRIES:
                raise
            wait = _RETRY_BACKOFF_BASE * (2 ** (attempt - 1))
            _logger.warning(
                '{"step":"tmdb_retry","status":"%s","attempt":%d,"wait_s":%.1f}',
                type(exc).__name__,
                attempt,
                wait,
            )
            time.sleep(wait)
            continue
        if resp.status_code in (429, 500, 502, 503, 504):
            attempt += 1
            if attempt > _MAX_RETRIES:
                resp.raise_for_status()
            retry_after = resp.headers.get("Retry-After")
            if retry_after is not None:
                wait = float(retry_after)
            else:
                wait = _RETRY_BACKOFF_BASE * (2 ** (attempt - 1))
            _logger.warning(
                '{"step":"tmdb_retry","status":%d,"attempt":%d,"wait_s":%.1f}',
                resp.status_code,
                attempt,
                wait,
            )
            time.sleep(wait)
            continue
        # Non-retryable error — let caller decide (raise_for_status or check code).
        time.sleep(FETCH_PACE_SECONDS)
        return resp


def discover_candidate_tmdb_ids(
    api_key: str,
    *,
    pages: int = 5,
    genre_ids: list[int] = DISCOVERY_GENRES,
) -> list[int]:
    ids: set[int] = set()

    for page in range(1, pages + 1):
        resp = tmdb_get(
            f"{TMDB_BASE}/movie/popular",
            api_key=api_key,
            params={"page": page},
        )
        resp.raise_for_status()
        for r in resp.json().get("results", []):
            ids.add(r["id"])

    for genre_id in genre_ids:
        for page in range(1, pages + 1):
            resp = tmdb_get(
                f"{TMDB_BASE}/discover/movie",
                api_key=api_key,
                params={
                    "sort_by": "popularity.desc",
                    "with_genres": genre_id,
                    "page": page,
                },
            )
            resp.raise_for_status()
            for r in resp.json().get("results", []):
                ids.add(r["id"])

    return list(ids)


def fetch_movie_metadata(
    tmdb_id: int, api_key: str, *, embed_text_recipe: str = "base"
) -> Optional[TmdbMovieMetadata]:
    resp = tmdb_get(
        f"{TMDB_BASE}/movie/{tmdb_id}",
        api_key=api_key,
        params={"append_to_response": "credits,keywords"},
    )
    if resp.status_code != 200:
        return None
    data = resp.json()

    genres = [g["name"] for g in data.get("genres", [])]
    cast = [c["name"] for c in data.get("credits", {}).get("cast", [])[:15]]
    director = next(
        (c["name"] for c in data.get("credits", {}).get("crew", []) if c["job"] == "Director"),
        "",
    )
    release_year = int((data.get("release_date") or "0000")[:4] or 0)
    keywords = [k["name"] for k in data.get("keywords", {}).get("keywords", [])]

    metadata = TmdbMovieMetadata(
        tmdb_id=tmdb_id,
        title=data.get("title", ""),
        year=release_year,
        overview=data.get("overview", ""),
        tagline=data.get("tagline", ""),
        genres=genres,
        cast=cast,
        director=director,
        runtime=data.get("runtime") or 0,
        vote_average=data.get("vote_average", 0.0),
        popularity=data.get("popularity", 0.0),
        keywords=keywords,
        embed_text="",
    )
    metadata.embed_text = build_movie_embed_text(metadata, recipe=embed_text_recipe)
    return metadata


def _existing_tmdb_ids(client: QdrantClient, collection_name: str) -> set[int]:
    """Return the set of tmdb_ids already stored in *collection_name*.

    Paginates via ``client.scroll`` following ``next_page_offset`` until ``None``
    so collections larger than one scroll page are fully enumerated (AC2).
    Only the ``tmdb_id`` payload field is requested to keep pages small.
    """
    existing: set[int] = set()
    offset: object = None  # initial offset — Qdrant accepts None to start from beginning
    while True:
        records, next_offset = client.scroll(
            collection_name=collection_name,
            limit=_SCROLL_PAGE_SIZE,
            with_payload=["tmdb_id"],
            with_vectors=False,
            offset=offset,
        )
        for record in records:
            tmdb_id = (record.payload or {}).get("tmdb_id")
            if tmdb_id is not None:
                existing.add(int(tmdb_id))
        if next_offset is None:
            break
        offset = next_offset
    return existing


def _process_movie(
    tmdb_id: int,
    *,
    api_key: str,
    embedder: Embedder,
    client: QdrantClient,
    collection_name: str,
    embed_text_recipe: str,
    sparse: bool,
) -> bool:
    """Fetch, embed, and upsert a single movie.  Returns True on success.

    ``metadata is None`` (TMDB fetch miss) returns False immediately.
    Any other exception propagates to the caller (caught in the ThreadPoolExecutor
    consumer as a per-item skip).

    The point id (``uuid5(NAMESPACE_DNS, str(tmdb_id))``) and payload dict shape
    are preserved exactly as before (AC7).
    """
    metadata = fetch_movie_metadata(tmdb_id, api_key, embed_text_recipe=embed_text_recipe)
    if metadata is None:
        return False

    # embed_texts is the document path; embed_single is reserved for queries
    # (it may prepend a query instruction, e.g. for bge models).
    dense_vector = embedder.embed_texts([metadata.embed_text])[0]
    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(tmdb_id)))

    if sparse:
        # Sample+sparse path: named dense vector ("") + BM25 Document ("text").
        # The client tokenises the Document locally via fastembed; the server
        # stores the sparse vector and applies IDF at query time.
        # Sparse text is built via the shared build_sparse_text (enriched-base
        # recipe: title, year, genres, director, cast top-5, tagline, overview,
        # keywords).  Using the shared builder here and in the backfill script
        # is the drift guard — both call sites are bound to the same function
        # so the recipes cannot diverge silently.
        _sparse_text = build_sparse_text(
            title=metadata.title,
            year=metadata.year,
            genres=metadata.genres,
            director=metadata.director,
            cast=metadata.cast,
            tagline=metadata.tagline,
            overview=metadata.overview,
            keywords=metadata.keywords,
        )
        vector: dict | list = {
            "": dense_vector,
            "text": models.Document(text=_sparse_text, model="Qdrant/bm25"),
        }
    else:
        vector = dense_vector

    client.upsert(
        collection_name=collection_name,
        points=[
            PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "tmdb_id": metadata.tmdb_id,
                    "title": metadata.title,
                    "year": metadata.year,
                    "genres": metadata.genres,
                    "cast": metadata.cast,
                    "director": metadata.director,
                    "overview": metadata.overview,
                    "tagline": metadata.tagline,
                    "runtime": metadata.runtime,
                    "vote_average": metadata.vote_average,
                    "popularity": metadata.popularity,
                    "themes": metadata.themes,
                    "keywords": metadata.keywords,
                },
            )
        ],
    )
    return True


def load_tmdb_movies(
    api_key: str,
    qdrant_url: str,
    qdrant_api_key: str,
    watched_tmdb_ids: set[int],
    embedder: Embedder,
    collection_name: str,
    *,
    discovery_pages: int = 5,
    genre_ids: list[int] = DISCOVERY_GENRES,
    explicit_tmdb_ids: list[int] | None = None,
    embed_text_recipe: str = "base",
    sparse: bool = False,
    resume: bool = False,
    workers: int = 8,
) -> int:
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=30)

    if explicit_tmdb_ids is not None:
        # Calibration sample: ingest exactly this fixed id list, no discovery.
        candidate_ids = list(explicit_tmdb_ids)
    else:
        candidate_ids = discover_candidate_tmdb_ids(
            api_key, pages=discovery_pages, genre_ids=genre_ids
        )
        candidate_ids = [i for i in candidate_ids if i not in watched_tmdb_ids]

    # Resume: skip ids already present in the collection (applies to both paths).
    if resume:
        existing = _existing_tmdb_ids(client, collection_name)
        before = len(candidate_ids)
        candidate_ids = [i for i in candidate_ids if i not in existing]
        _logger.info(
            '{"step":"tmdb_movies_resume","existing":%d,"skipped":%d,"remaining":%d}',
            len(existing),
            before - len(candidate_ids),
            len(candidate_ids),
        )

    _logger.info('{"step":"tmdb_movies_candidates","count":%d}', len(candidate_ids))

    loaded = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_id = {
            executor.submit(
                _process_movie,
                tmdb_id,
                api_key=api_key,
                embedder=embedder,
                client=client,
                collection_name=collection_name,
                embed_text_recipe=embed_text_recipe,
                sparse=sparse,
            ): tmdb_id
            for tmdb_id in candidate_ids
        }
        for future in as_completed(future_to_id):
            tmdb_id = future_to_id[future]
            try:
                if future.result():
                    loaded += 1
            except Exception as exc:
                _logger.warning(
                    '{"step":"movie_skip","tmdb_id":%d,"error":"%s"}',
                    tmdb_id,
                    type(exc).__name__,
                )

    _logger.info('{"step":"tmdb_movies_done","loaded":%d}', loaded)
    return loaded
