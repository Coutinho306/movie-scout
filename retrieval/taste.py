"""Re-score MovieHits against the user's taste centroid.

Callers must supply the profile explicitly. ``profile=None`` is handled at the
call site (``match_taste_tool``) as cold start — these functions require a
non-None profile. The ``_TASTE_PROFILE_PATH`` constant and
``load_taste_profile`` are retained for the offline script and tests only;
they are **not** used on the serving path.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ingestion.models import TasteProfile
from ingestion.scripts.compute_taste import load_taste_profile
from retrieval.models import MovieHit

if TYPE_CHECKING:
    from retrieval.config import RetrievalSettings

# Retained for offline script + tests only. NOT loaded on the serving path.
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
    profile: TasteProfile,
    alpha: float = 0.5,
) -> list[MovieHit]:
    """No-vector fallback: rank by retrieval score, taste ignored.

    Use this only when per-hit embedding vectors are unavailable. With no
    vectors there is nothing to compare against the centroid, so taste_score is
    0 and blended collapses to ``alpha * retrieval_score`` — the ordering is the
    retrieval ordering, scaled. When vectors *are* available, callers must use
    :func:`score_against_taste_with_vectors` instead for real taste ranking.

    ``profile`` must be supplied by the caller (no implicit file load on the
    serving path). Returns hits sorted descending by blended_score.
    """
    if not hits:
        return hits

    updated: list[MovieHit] = []
    for hit in hits:
        taste_score = 0.0  # no vector to compare against the centroid
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
    profile: TasteProfile,
    alpha: float = 0.5,
) -> list[MovieHit]:
    """Variant when caller supplies the embedding vectors for each hit.

    vectors[i] must correspond to hits[i].
    ``profile`` must be supplied by the caller (no implicit file load on the
    serving path).
    """
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
