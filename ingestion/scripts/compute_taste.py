"""Script: read Letterboxd CSVs → embed → compute taste centroids → save taste_profile.json."""

import difflib
import json
import logging
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from dotenv import load_dotenv

from ingestion.chunking import build_movie_embed_text
from ingestion.embedding import embed_texts
from ingestion.models import (
    LetterboxdFilm,
    TasteProfile,
    TmdbMovieMetadata,
    TmdbSearchResult,
)

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(message)s")
_logger = logging.getLogger(__name__)

TMDB_BASE = "https://api.themoviedb.org/3"


def load_letterboxd_csvs(
    export_dir: Path,
) -> tuple[list[LetterboxdFilm], list[LetterboxdFilm]]:
    pool: list[LetterboxdFilm] = []
    seen: set[tuple[str, int]] = set()

    ratings_df = pd.read_csv(export_dir / "ratings.csv")
    for _, row in ratings_df.iterrows():
        key = (row["Name"], int(row["Year"]))
        if key not in seen:
            pool.append(
                LetterboxdFilm(
                    name=row["Name"],
                    year=int(row["Year"]),
                    rating=float(row["Rating"]),
                    source="rated",
                )
            )
            seen.add(key)

    liked_df = pd.read_csv(export_dir / "likes" / "films.csv")
    for _, row in liked_df.iterrows():
        key = (row["Name"], int(row["Year"]))
        if key not in seen:
            pool.append(
                LetterboxdFilm(
                    name=row["Name"], year=int(row["Year"]), rating=None, source="liked"
                )
            )
            seen.add(key)

    watched_df = pd.read_csv(export_dir / "watched.csv")
    for _, row in watched_df.iterrows():
        key = (row["Name"], int(row["Year"]))
        if key not in seen:
            pool.append(
                LetterboxdFilm(
                    name=row["Name"],
                    year=int(row["Year"]),
                    rating=None,
                    source="watched",
                )
            )
            seen.add(key)

    watchlist_df = pd.read_csv(export_dir / "watchlist.csv")
    watchlist = [
        LetterboxdFilm(
            name=row["Name"], year=int(row["Year"]), rating=None, source="watchlist"
        )
        for _, row in watchlist_df.iterrows()
    ]

    return pool, watchlist


def search_tmdb(
    name: str, year: int, api_key: str
) -> Optional[TmdbSearchResult]:
    resp = requests.get(
        f"{TMDB_BASE}/search/movie",
        params={"query": name, "year": year},
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=10,
    )
    resp.raise_for_status()
    results = resp.json().get("results", [])
    best: Optional[TmdbSearchResult] = None
    for r in results:
        release_year = int((r.get("release_date") or "0000")[:4] or 0)
        if abs(release_year - year) > 1:
            continue
        score = difflib.SequenceMatcher(None, name.lower(), r["title"].lower()).ratio()
        if score > 0.8 and (best is None or score > best.match_score):
            best = TmdbSearchResult(
                tmdb_id=r["id"],
                title=r["title"],
                year=release_year,
                match_score=score,
            )
    return best


def fetch_tmdb_metadata_for_taste(
    tmdb_id: int, api_key: str
) -> Optional[TmdbMovieMetadata]:
    resp = requests.get(
        f"{TMDB_BASE}/movie/{tmdb_id}",
        params={"append_to_response": "credits"},
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=10,
    )
    if resp.status_code != 200:
        return None
    data = resp.json()

    genres = [g["name"] for g in data.get("genres", [])]
    cast = [
        c["name"]
        for c in data.get("credits", {}).get("cast", [])[:5]
    ]
    director = next(
        (
            c["name"]
            for c in data.get("credits", {}).get("crew", [])
            if c["job"] == "Director"
        ),
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


def rating_weight(film: LetterboxdFilm) -> float:
    if film.source == "rated" and film.rating is not None:
        mapping = {5.0: 1.0, 4.5: 0.9, 4.0: 0.8, 3.5: 0.65, 3.0: 0.5, 2.5: 0.35, 2.0: 0.2, 1.5: 0.1, 1.0: 0.0}
        return mapping.get(film.rating, 0.0)
    if film.source == "liked":
        return 0.7
    if film.source == "watched":
        return 0.3
    return 0.0


def compute_centroid(
    vectors: list[list[float]], weights: list[float]
) -> list[float]:
    dim = len(vectors[0])
    centroid = [0.0] * dim
    total_weight = sum(weights)
    for vec, w in zip(vectors, weights):
        for i in range(dim):
            centroid[i] += vec[i] * w / total_weight
    norm = math.sqrt(sum(x * x for x in centroid))
    return [x / norm for x in centroid]


def main() -> None:
    api_key = os.environ["TMDB_API_KEY"]
    export_dir = Path("data/letterboxd_export")
    output_path = Path("data/taste_profile.json")

    _logger.info('{"step":"load_csvs"}')
    pool, _ = load_letterboxd_csvs(export_dir)
    _logger.info('{"step":"loaded","film_count":%d}', len(pool))

    weighted_films = [(f, rating_weight(f)) for f in pool]
    weighted_films = [(f, w) for f, w in weighted_films if w > 0.0]

    texts_to_embed: list[str] = []
    metadata_list: list[TmdbMovieMetadata] = []
    weights: list[float] = []

    for film, weight in weighted_films:
        _logger.info('{"step":"search_tmdb","film":"%s","year":%d}', film.name, film.year)
        result = search_tmdb(film.name, film.year, api_key)
        time.sleep(0.25)
        if result is None:
            _logger.info('{"step":"tmdb_miss","film":"%s"}', film.name)
            continue

        metadata = fetch_tmdb_metadata_for_taste(result.tmdb_id, api_key)
        time.sleep(0.25)
        if metadata is None:
            continue

        texts_to_embed.append(metadata.embed_text)
        metadata_list.append(metadata)
        weights.append(weight)

    _logger.info('{"step":"embedding","count":%d}', len(texts_to_embed))
    vectors = embed_texts(texts_to_embed)

    centroid = compute_centroid(vectors, weights)

    rated = sum(1 for f, _ in weighted_films if f.source == "rated")
    liked = sum(1 for f, _ in weighted_films if f.source == "liked")

    profile = TasteProfile(
        centroid=centroid,
        film_count=len(vectors),
        rated_count=rated,
        liked_count=liked,
        created_at=datetime.now(timezone.utc).isoformat(),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(profile.model_dump(), indent=2))
    _logger.info(
        '{"step":"done","film_count":%d,"centroid_dim":%d,"output":"%s"}',
        profile.film_count,
        len(centroid),
        str(output_path),
    )


if __name__ == "__main__":
    main()
