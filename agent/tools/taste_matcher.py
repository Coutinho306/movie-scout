"""Tool: cosine similarity of candidate vs taste centroid.

``profile=None`` means cold start — no taste profile uploaded this session.
In cold-start mode, hits are returned in retrieval order (no re-ranking).
"""

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
    """Re-rank hits by taste centroid cosine similarity.

    When ``profile`` is None (cold start — no upload this session), hits are
    returned in their original retrieval order with no re-ranking. This is the
    unambiguous cold-start path; it does not rely on alpha arithmetic.

    When ``profile`` is provided and every hit carries its embedding vector,
    uses the real centroid cosine; otherwise falls back to retrieval order.
    """
    if profile is None:
        # Cold start: no profile uploaded — return retrieval order unchanged.
        logger.debug('{"step":"taste_cold_start","reason":"no profile"}')
        return list(hits)

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
