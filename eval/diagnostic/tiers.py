"""Tier query construction for the retrieval diagnostic suite.

Four difficulty tiers derived mechanically from stored Qdrant payloads
(tiers 0-2) and the existing cached LLM-generated query (tier 3).  No I/O
here: all data is passed in; this module is pure functions + Pydantic models.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

PopularityTier = Literal["popular", "mid", "niche"]
ReviewCoverage = Literal["reviews", "no_reviews"]


class TierQuery(BaseModel):
    """A single (tier, target) query entry in the diagnostic suite."""

    text: str
    target_tmdb_ids: set[int]
    tier: int  # 0, 1, 2, or 3
    popularity_tier: PopularityTier
    review_coverage: ReviewCoverage


class DiagnosticSuite(BaseModel):
    """The full 120-query diagnostic suite (30 films × 4 tiers)."""

    queries: list[TierQuery]


def _tier0(payload: dict) -> str:
    """Tier 0: verbatim title."""
    return payload["title"]


def _tier1(payload: dict) -> str:
    """Tier 1: first sentence of overview (fallback to title if empty)."""
    overview = (payload.get("overview") or "").strip()
    if not overview:
        return payload["title"]
    first_sentence, _, _ = overview.partition(". ")
    return first_sentence.strip()


def _tier2(payload: dict) -> str:
    """Tier 2: deterministic genre+mood synthesis.

    Template: ``"a {genres joined} film — {tagline or first 8 overview words}"``.
    """
    genres_raw = payload.get("genres") or []
    genres = ", ".join(genres_raw) if genres_raw else "film"
    tagline = (payload.get("tagline") or "").strip()
    if tagline:
        mood = tagline
    else:
        overview = (payload.get("overview") or "").strip()
        mood = " ".join(overview.split()[:8])
    return f"a {genres} film — {mood}"


def build_tier_queries(
    payload: dict,
    *,
    seed_tmdb_id: int,
    target_tmdb_ids: set[int],
    tier3_text: str,
    popularity_tier: PopularityTier,
    review_coverage: ReviewCoverage,
) -> list[TierQuery]:
    """Return the four TierQuery objects for one target film.

    Parameters
    ----------
    payload:
        The seed film's stored Qdrant payload dict (must have at minimum
        ``title``, ``overview``, ``genres``, ``tagline``). Used for tier 0-2
        query text construction only.
    seed_tmdb_id:
        The seed film's TMDB id. Used for payload fetch / tier-0/1/2 text;
        distinct from the full relevance set carried in ``target_tmdb_ids``.
    target_tmdb_ids:
        The full set of relevant TMDB ids for this query (seed + cluster
        members). Carried through to every TierQuery unchanged.
    tier3_text:
        The pre-cached LLM-generated abstract query for this film.
    popularity_tier:
        The film's popularity bucket (``popular`` / ``mid`` / ``niche``),
        re-derived by the caller from corpus percentiles.
    review_coverage:
        Whether the film has any review chunks in ``tmdb_reviews``.
    """
    common = dict(
        target_tmdb_ids=target_tmdb_ids,
        popularity_tier=popularity_tier,
        review_coverage=review_coverage,
    )
    return [
        TierQuery(text=_tier0(payload), tier=0, **common),
        TierQuery(text=_tier1(payload), tier=1, **common),
        TierQuery(text=_tier2(payload), tier=2, **common),
        TierQuery(text=tier3_text, tier=3, **common),
    ]
