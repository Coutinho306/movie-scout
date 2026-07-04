"""Domain models for retrieval results and filters."""

from __future__ import annotations

from pydantic import BaseModel, Field


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
    # Dense embedding, populated when with_vectors=True. exclude=True keeps it
    # out of every model_dump()/json() — it's for internal scoring math only,
    # and ~1536 floats serialized into an LLM tool result blows the context.
    vector: list[float] | None = Field(default=None, exclude=True)


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
    cast: list[str] | None = None
    vote_min: float | None = None
    exclude_tmdb_ids: set[int] | None = None
