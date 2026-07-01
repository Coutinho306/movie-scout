"""Tool: cosine similarity of candidate vs taste centroids loaded from taste_profile.json."""

import logging

from ingestion.models import TasteProfile
from retrieval.models import MovieHit
from retrieval.taste import score_against_taste, score_against_taste_with_vectors

logger = logging.getLogger(__name__)


def match_taste_tool(
    hits: list[MovieHit],
    *,
    profile: TasteProfile | None = None,
    alpha: float = 0.5,
) -> list[MovieHit]:
    """Re-rank hits by taste. Uses the real centroid cosine when every hit
    carries its embedding vector; falls back to retrieval-only order otherwise.
    """
    if hits and all(h.vector is not None for h in hits):
        vectors = [h.vector for h in hits]  # all non-None per the guard above
        return score_against_taste_with_vectors(
            hits, vectors, profile=profile, alpha=alpha  # type: ignore[arg-type]
        )
    if hits:
        logger.warning(
            '{"step":"taste_fallback","reason":"missing hit vectors, taste ignored"}'
        )
    return score_against_taste(hits, profile=profile, alpha=alpha)
