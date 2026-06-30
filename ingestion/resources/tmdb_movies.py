"""TMDB candidate pool — fetch metadata, embed, load Qdrant."""

import logging
import time
import uuid
from typing import Optional

import requests
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from ingestion.chunking import build_movie_embed_text
from ingestion.embedding import Embedder
from ingestion.models import TmdbMovieMetadata

_logger = logging.getLogger(__name__)
TMDB_BASE = "https://api.themoviedb.org/3"
DISCOVERY_GENRES = [18, 28, 53]  # Drama, Action, Thriller


def discover_candidate_tmdb_ids(
    api_key: str,
    *,
    pages: int = 5,
    genre_ids: list[int] = DISCOVERY_GENRES,
) -> list[int]:
    ids: set[int] = set()

    for page in range(1, pages + 1):
        resp = requests.get(
            f"{TMDB_BASE}/movie/popular",
            params={"page": page},
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10,
        )
        resp.raise_for_status()
        for r in resp.json().get("results", []):
            ids.add(r["id"])
        time.sleep(0.25)

    for genre_id in genre_ids:
        for page in range(1, pages + 1):
            resp = requests.get(
                f"{TMDB_BASE}/discover/movie",
                params={
                    "sort_by": "popularity.desc",
                    "with_genres": genre_id,
                    "page": page,
                },
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=10,
            )
            resp.raise_for_status()
            for r in resp.json().get("results", []):
                ids.add(r["id"])
            time.sleep(0.25)

    return list(ids)


def fetch_movie_metadata(
    tmdb_id: int, api_key: str
) -> Optional[TmdbMovieMetadata]:
    resp = requests.get(
        f"{TMDB_BASE}/movie/{tmdb_id}",
        params={"append_to_response": "credits,keywords"},
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=10,
    )
    if resp.status_code != 200:
        return None
    data = resp.json()

    genres = [g["name"] for g in data.get("genres", [])]
    cast = [c["name"] for c in data.get("credits", {}).get("cast", [])[:5]]
    director = next(
        (c["name"] for c in data.get("credits", {}).get("crew", []) if c["job"] == "Director"),
        "",
    )
    release_year = int((data.get("release_date") or "0000")[:4] or 0)

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
        embed_text="",
    )
    metadata.embed_text = build_movie_embed_text(metadata)
    return metadata


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
) -> int:
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    candidate_ids = discover_candidate_tmdb_ids(
        api_key, pages=discovery_pages, genre_ids=genre_ids
    )
    candidate_ids = [i for i in candidate_ids if i not in watched_tmdb_ids]
    _logger.info('{"step":"tmdb_movies_candidates","count":%d}', len(candidate_ids))

    loaded = 0
    for tmdb_id in candidate_ids:
        metadata = fetch_movie_metadata(tmdb_id, api_key)
        time.sleep(0.25)
        if metadata is None:
            continue

        vector = embedder.embed_single(metadata.embed_text)
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(tmdb_id)))

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
                    },
                )
            ],
        )
        loaded += 1

    _logger.info('{"step":"tmdb_movies_done","loaded":%d}', loaded)
    return loaded
