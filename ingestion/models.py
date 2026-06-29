from typing import Optional

from pydantic import BaseModel


class LetterboxdFilm(BaseModel):
    name: str
    year: int
    rating: Optional[float]
    source: str  # "rated" | "watched" | "liked" | "watchlist"


class TmdbSearchResult(BaseModel):
    tmdb_id: int
    title: str
    year: int
    match_score: float


class TmdbMovieMetadata(BaseModel):
    tmdb_id: int
    title: str
    year: int
    overview: str
    tagline: str
    genres: list[str]
    cast: list[str]
    director: str
    runtime: int
    vote_average: float
    popularity: float
    embed_text: str


class TmdbReviewChunk(BaseModel):
    tmdb_id: int
    title: str
    review_author: str
    chunk_index: int
    chunk_text: str
    total_chunks: int


class TasteProfile(BaseModel):
    centroid: list[float]
    film_count: int
    rated_count: int
    liked_count: int
    created_at: str
