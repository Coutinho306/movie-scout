"""Tool: hybrid vector search over tmdb_reviews Qdrant collection."""

from retrieval.config import RetrievalSettings
from retrieval.models import ReviewHit
from retrieval.reviews import search_reviews


def search_reviews_tool(
    query: str,
    *,
    settings: RetrievalSettings | None = None,
    k: int = 10,
    tmdb_ids: list[int] | None = None,
) -> list[ReviewHit]:
    return search_reviews(query, settings=settings or RetrievalSettings(), k=k, tmdb_ids=tmdb_ids)
