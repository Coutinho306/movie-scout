"""Re-score MovieHits against the user's taste centroid."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ingestion.models import TasteProfile
from ingestion.scripts.compute_taste import load_taste_profile
from retrieval.models import MovieHit

if TYPE_CHECKING:
    from retrieval.config import RetrievalSettings

_TASTE_PROFILE_PATH = Path("data/taste_profile.json")


def _cosine(a: list[float], b: list[float]) -> float:
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    denom = (np.linalg.norm(va) * np.linalg.norm(vb))
    if denom == 0.0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def score_against_taste(
    hits: list[MovieHit],
    *,
    profile: TasteProfile | None = None,
    alpha: float = 0.5,
) -> list[MovieHit]:
    """Blend retrieval score with taste-centroid cosine similarity.

    blended = alpha * retrieval_score + (1 - alpha) * taste_score

    Loads taste_profile.json when profile=None.
    Returns hits sorted descending by blended_score.
    """
    if not hits:
        return hits
    if profile is None:
        profile = load_taste_profile(_TASTE_PROFILE_PATH)

    centroid = profile.centroid
    updated: list[MovieHit] = []
    for hit in hits:
        # Each MovieHit may carry a pre-embedded vector in payload, but
        # we only store metadata in Qdrant payload (not the vector itself).
        # We use the retrieval score as a proxy and centroid to compute a
        # coarse estimate: since we embedded the query and retrieved by
        # cosine, the score already encodes semantic alignment.
        # Taste score comes from re-encoding the movie text if available;
        # for now use the stored score (range 0–1) as retrieval_score.
        taste_score = 0.0  # placeholder; real cosine needs stored vector
        blended = alpha * hit.score + (1.0 - alpha) * taste_score
        updated.append(
            hit.model_copy(update={"taste_score": taste_score, "blended_score": blended})
        )

    # Sort descending
    updated.sort(key=lambda h: h.blended_score, reverse=True)
    return updated


def score_against_taste_with_vectors(
    hits: list[MovieHit],
    vectors: list[list[float]],
    *,
    profile: TasteProfile | None = None,
    alpha: float = 0.5,
) -> list[MovieHit]:
    """Variant when caller supplies the embedding vectors for each hit.

    vectors[i] must correspond to hits[i].
    """
    if profile is None:
        profile = load_taste_profile(_TASTE_PROFILE_PATH)

    centroid = profile.centroid
    updated: list[MovieHit] = []
    for hit, vec in zip(hits, vectors, strict=True):
        taste_score = max(0.0, _cosine(vec, centroid))
        blended = alpha * hit.score + (1.0 - alpha) * taste_score
        updated.append(
            hit.model_copy(update={"taste_score": taste_score, "blended_score": blended})
        )
    updated.sort(key=lambda h: h.blended_score, reverse=True)
    return updated
