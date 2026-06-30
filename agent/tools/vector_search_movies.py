"""Tool: hybrid vector search over tmdb_movies Qdrant collection."""

from retrieval.config import RetrievalSettings
from retrieval.models import MovieFilters, MovieHit
from retrieval.movies import search_movies


def search_movies_tool(
    query: str,
    *,
    settings: RetrievalSettings | None = None,
    k: int = 10,
    filters: MovieFilters | None = None,
) -> list[MovieHit]:
    return search_movies(query, settings=settings or RetrievalSettings(), k=k, filters=filters)
