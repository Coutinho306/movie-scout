"""Franchise-ambiguity detection for seed-shaped queries.

When a user asks "films like X" and X belongs to a franchise with other
members already in the corpus, the intent is ambiguous — do they want
sequels/prequels included, or just films with a similar vibe?

This module:
- Detects that ambiguity deterministically (no LLM call).
- Builds a templated clarification question.
- Parses the user's free-text answer into a tri-state bool.
"""

from __future__ import annotations

import logging
import uuid

import requests
from pydantic import BaseModel

from agent.tools.seed_film import extract_seed_title
from agent.tools.tmdb_search import _BASE_URL, search_tmdb
from retrieval.client import get_qdrant_client

logger = logging.getLogger(__name__)

_COLLECTION = "tmdb_movies"


def _point_id(tmdb_id: int) -> str:
    """Derive the Qdrant point id from a TMDB movie id (mirrors ingestion)."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, str(tmdb_id)))


class FranchiseAmbiguity(BaseModel):
    """Detected franchise ambiguity for a seed-shaped query."""

    seed_id: int
    seed_title: str
    collection_name: str
    sibling_ids: list[int]  # other franchise members present in corpus (excl. seed)
    question: str  # ready-to-display clarification question


def _fetch_movie_details(tmdb_id: int, tmdb_api_key: str) -> dict | None:
    """Fetch TMDB /movie/{id}?append_to_response=belongs_to_collection."""
    try:
        resp = requests.get(
            f"{_BASE_URL}/movie/{tmdb_id}",
            headers={"Authorization": f"Bearer {tmdb_api_key}", "accept": "application/json"},
            params={"append_to_response": "belongs_to_collection"},
            timeout=10.0,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        logger.warning("TMDB movie detail fetch failed for id=%d: %s", tmdb_id, exc)
        return None


def _fetch_collection_members(collection_id: int, tmdb_api_key: str) -> list[int]:
    """Fetch TMDB /collection/{id} and return all member tmdb_ids."""
    try:
        resp = requests.get(
            f"{_BASE_URL}/collection/{collection_id}",
            headers={"Authorization": f"Bearer {tmdb_api_key}", "accept": "application/json"},
            timeout=10.0,
        )
        resp.raise_for_status()
        parts = resp.json().get("parts", [])
        return [p["id"] for p in parts if "id" in p]
    except requests.RequestException as exc:
        logger.warning("TMDB collection fetch failed for id=%d: %s", collection_id, exc)
        return []


def _corpus_members(tmdb_ids: list[int]) -> list[int]:
    """Return the subset of tmdb_ids that exist as points in the corpus."""
    if not tmdb_ids:
        return []
    point_ids = [_point_id(tid) for tid in tmdb_ids]
    try:
        client = get_qdrant_client()
        records = client.retrieve(
            collection_name=_COLLECTION,
            ids=point_ids,
            with_payload=False,
            with_vectors=False,
        )
    except Exception as exc:  # noqa: BLE001 — Qdrant unavailable must not crash
        logger.warning("Qdrant corpus-membership check failed: %s", exc)
        return []

    # Map back from point id strings to tmdb_ids
    present_point_ids: set[str] = {str(r.id) for r in records}
    return [tid for tid, pid in zip(tmdb_ids, point_ids) if pid in present_point_ids]


def _build_question(seed_title: str, collection_name: str, genres: list[str]) -> str:
    """Build the clarification question, optionally enriched with genre words."""
    if genres:
        # Use up to two genre names for context
        genre_phrase = "/".join(genres[:2]).lower()
        vibe = f"similar {genre_phrase} vibe"
    else:
        vibe = "similar vibe"
    return (
        f'"{seed_title}" is part of the {collection_name} — do you want those included, '
        f"or just films with a {vibe}? (yes / no)"
    )


def detect_franchise_ambiguity(
    query: str,
    *,
    tmdb_api_key: str,
) -> FranchiseAmbiguity | None:
    """Return a FranchiseAmbiguity if the query is seed-shaped and the seed belongs
    to a franchise with sibling members in the corpus. Otherwise returns None.

    Detection steps:
    1. Extract seed title — non-seed queries return None immediately (no TMDB call).
    2. Resolve seed to a tmdb_id via TMDB search.
    3. Fetch /movie/{id}?append_to_response=belongs_to_collection.
    4. If the film has a collection, fetch all collection member ids.
    5. Filter to corpus members (excluding the seed itself).
    6. If any siblings are in corpus → return FranchiseAmbiguity; else None.
    """
    # Step 1: only fire on seed-shaped queries
    seed_title = extract_seed_title(query)
    if seed_title is None:
        return None

    # Step 2: resolve seed to tmdb_id
    seed_id = search_tmdb(seed_title)
    if seed_id is None:
        return None

    # Step 3: fetch movie details + belongs_to_collection
    details = _fetch_movie_details(seed_id, tmdb_api_key)
    if details is None:
        return None

    collection_info = details.get("belongs_to_collection")
    if not collection_info:
        return None

    collection_id: int = collection_info["id"]
    collection_name: str = collection_info.get("name", "this franchise")

    # Step 4: fetch all collection member tmdb_ids
    all_member_ids = _fetch_collection_members(collection_id, tmdb_api_key)

    # Step 5: filter to sibling corpus members (exclude the seed itself)
    sibling_candidates = [mid for mid in all_member_ids if mid != seed_id]
    corpus_siblings = _corpus_members(sibling_candidates)

    # Step 6: no corpus siblings → no ambiguity
    if not corpus_siblings:
        return None

    # Optionally enrich question with seed's genre names (SPEC open question, non-load-bearing)
    genres = [g.get("name", "") for g in details.get("genres", []) if g.get("name")]

    question = _build_question(seed_title, collection_name, genres)

    return FranchiseAmbiguity(
        seed_id=seed_id,
        seed_title=seed_title,
        collection_name=collection_name,
        sibling_ids=corpus_siblings,
        question=question,
    )


# ---------------------------------------------------------------------------
# Clarification answer parser
# ---------------------------------------------------------------------------

_AFFIRMATIVE_TOKENS = frozenset({
    "yes", "yeah", "yep", "yup", "sure", "ok", "okay", "include", "included",
    "all", "both", "too", "also",
    "absolutely", "definitely", "please", "y",
})

# Strong negative signals — presence of any of these leans negative
# even if other tokens are affirmative-adjacent (e.g. "exclude them",
# "skip the sequels" — "them"/"sequels" are just referring words, not affirming)
_NEGATIVE_TOKENS = frozenset({
    "no", "nope", "nah", "not", "without", "exclude", "excluded", "skip",
    "just", "only", "vibe", "similar", "different", "none", "n",
    "avoid", "avoiding", "excluding", "omit", "omitting",
})

# Tokens that are affirmative-adjacent but act as references; they should
# not count as affirmative when the overall signal is negative.
_REFERENCE_ONLY_TOKENS = frozenset({"them", "those", "with"})


def resolve_clarification(answer: str) -> bool | None:
    """Parse a free-text clarification answer into a tri-state bool.

    Returns:
        True   — user wants franchise siblings included.
        False  — user wants only non-franchise "vibe" films.
        None   — unclear / off-topic; caller should apply the default (exclude).
    """
    normalized = answer.lower().strip()
    if not normalized:
        return None

    tokens = set(normalized.split())

    neg_hits = tokens & _NEGATIVE_TOKENS
    # Affirmative hits excluding reference-only words (they don't signal
    # affirmation when standing alone alongside negative tokens)
    strong_aff_hits = tokens & _AFFIRMATIVE_TOKENS

    if neg_hits:
        # Negative signal present: return False unless strong affirmatives
        # are also present, which makes it unclear.
        if strong_aff_hits:
            return None  # conflicting ("yes but no")
        return False

    if strong_aff_hits:
        return True

    # Nothing recognisable → unclear
    return None
