"""Tool: cosine similarity of candidate vs taste centroids loaded from taste_profile.json."""

from ingestion.models import TasteProfile
from retrieval.models import MovieHit
from retrieval.taste import score_against_taste, score_against_taste_with_vectors


def match_taste_tool(
    hits: list[MovieHit],
    *,
    profile: TasteProfile | None = None,
    alpha: float = 0.5,
) -> list[MovieHit]:
    return score_against_taste(hits, profile=profile, alpha=alpha)
