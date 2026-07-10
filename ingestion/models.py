from typing import Optional

from pydantic import BaseModel

# TMDB's standard movie genres are stable; hardcode the name→ID map rather than
# calling /genre/movie/list. Used to turn taste genre names into discover IDs.
TMDB_GENRE_NAME_TO_ID: dict[str, int] = {
    "Action": 28,
    "Adventure": 12,
    "Animation": 16,
    "Comedy": 35,
    "Crime": 80,
    "Documentary": 99,
    "Drama": 18,
    "Family": 10751,
    "Fantasy": 14,
    "History": 36,
    "Horror": 27,
    "Music": 10402,
    "Mystery": 9648,
    "Romance": 10749,
    "Science Fiction": 878,
    "TV Movie": 10770,
    "Thriller": 53,
    "War": 10752,
    "Western": 37,
}


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
    keywords: list[str] = []
    themes: list[str] = []  # reserved for feature C (LLM-extracted); empty for now
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
    top_genre_ids: list[int]
    genre_weights: dict[str, float]
    created_at: str
    top_films: list[str] = []
