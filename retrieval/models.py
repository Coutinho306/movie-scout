"""Domain models for retrieval results and filters."""

from __future__ import annotations

from pydantic import BaseModel


class MovieHit(BaseModel):
    tmdb_id: int
    title: str
    year: int
    overview: str
    genres: list[str]
    vote_average: float
    score: float  # vector similarity score
    taste_score: float = 0.0
    blended_score: float = 0.0
    vector: list[float] | None = None  # dense embedding, populated when with_vectors=True


class ReviewHit(BaseModel):
    tmdb_id: int
    title: str
    review_author: str
    chunk_text: str
    chunk_index: int
    score: float


class MovieFilters(BaseModel):
    year_min: int | None = None
    year_max: int | None = None
    genres: list[str] | None = None
    vote_min: float | None = None
    exclude_tmdb_ids: set[int] | None = None
