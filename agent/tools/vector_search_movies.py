"""Tool: hybrid vector search and point-similarity search over tmdb_movies.

search_movies_tool tries to resolve a named seed film deterministically
(regex extraction + TMDB lookup, no LLM call) before falling back to plain
text embedding. This removes the LLM tool-choice ambiguity that caused
inconsistent routing between a raw text search and a seed-similarity search
for the same query across separate runs (see specs/features/
agent-named-film-resolution/SPEC.md's manual validation notes).
"""

from agent.tools.seed_film import extract_seed_title
from agent.tools.tmdb_search import search_tmdb
from retrieval.config import RetrievalSettings
from retrieval.models import MovieFilters, MovieHit
from retrieval.movies import recommend_similar, search_movies


def search_movies_tool(
    query: str,
    *,
    settings: RetrievalSettings | None = None,
    k: int = 10,
    filters: MovieFilters | None = None,
) -> list[MovieHit]:
    seed_title = extract_seed_title(query)
    if seed_title:
        seed_tmdb_id = search_tmdb(seed_title)
        if seed_tmdb_id is not None:
            hits = similar_movies_tool(
                seed_tmdb_id, settings=settings, k=k, filters=filters
            )
            if hits:
                return hits
    return search_movies(query, settings=settings or RetrievalSettings(), k=k, filters=filters)


def similar_movies_tool(
    seed_tmdb_id: int,
    *,
    settings: RetrievalSettings | None = None,
    k: int = 10,
    filters: MovieFilters | None = None,
) -> list[MovieHit]:
    """Return movies similar to the seed film using point-to-point vector similarity.

    The seed's own tmdb_id is always excluded from results (self-exclusion).
    Returns [] when the seed is not found in the corpus.
    """
    effective_filters = filters or MovieFilters()
    # Ensure seed is always excluded, merging with any caller-supplied set.
    seed_exclude: set[int] = {seed_tmdb_id}
    if effective_filters.exclude_tmdb_ids:
        seed_exclude |= effective_filters.exclude_tmdb_ids
    effective_filters = effective_filters.model_copy(
        update={"exclude_tmdb_ids": seed_exclude}
    )
    return recommend_similar(
        seed_tmdb_id,
        settings=settings or RetrievalSettings(),
        k=k,
        filters=effective_filters,
    )
