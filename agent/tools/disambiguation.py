"""Title-collision disambiguation for same-named films.

Provides detection of exact-title collisions (e.g. four films all named
"Obsession") and resolution of a user's free-text year reference to a single
tmdb_id. Title extraction is regex-first (free, deterministic, covers common
phrasings); a cheap LLM call (``_extract_title_via_llm``) fires only as a
fallback — when regex can't parse the query, or regex extracted a title that
hits zero corpus results. The ~95% of queries that regex handles cleanly
never touch the LLM. Year resolution stays fully deterministic, no LLM call.

This module is the single source of truth for:
- ``extract_title_from_query`` (hoisted from ``agent/nodes/synthesize.py``'s
  private ``_extract_title_from_query``; synthesize.py now imports from here).
- ``_extract_title_via_llm`` — cheap-model fallback, only reached on regex miss.
- ``TitleCollision`` / ``CollisionCandidate`` pydantic models.
- ``detect_title_collision`` — pre-graph gate function (mirrors franchise.py's
  ``detect_franchise_ambiguity``).
- ``resolve_year_reference`` — pure, deterministic year-pick resolver.
- ``build_collision_question`` — templated clarification question (no LLM).
"""

from __future__ import annotations

import re
import uuid
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel
from retrieval.movies import find_by_exact_title  # noqa: E402 — retrieval has no dep on agent

_EXTRACT_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "extract_title.md"
_EXTRACT_PROMPT_TEMPLATE = _EXTRACT_PROMPT_PATH.read_text()
_EXTRACT_MODEL = "gpt-4o-mini"

# Maximum year distance (inclusive) for a fuzzy nearest-year match.
# Distance 1 resolves near-miss typos ("the 2025 one" → 2026 film).
# Distance > 1 (e.g. "the 1990 one" vs {1943,1976,2015,2026}: dist=14) is
# treated as unresolvable (returns None) — confirmed design decision 2026-07-09.
MAX_YEAR_DISTANCE: int = 1


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class CollisionCandidate(BaseModel):
    """A single film in a same-title collision set."""

    tmdb_id: int
    year: int


class TitleCollision(BaseModel):
    """A confirmed same-title collision: ≥2 films sharing the same title."""

    title: str
    candidates: list[CollisionCandidate]


# ---------------------------------------------------------------------------
# Title extraction  (single definition — imported by synthesize.py too)
# ---------------------------------------------------------------------------


def extract_title_from_query(query: str) -> str | None:
    """Heuristically extract a film title from an inform-intent query.

    Looks for a quoted title first, then strips common question prefixes to
    isolate a bare title. Strips a trailing 4-digit year (e.g. "Obsession
    2026" -> "Obsession") since exact-title lookup matches the bare title
    payload field, not "title year". Returns None when extraction is
    uncertain — callers treat that as "no collision lookup needed".
    """
    # Quoted title: "What is the theme of 'Obsession'?" -> "Obsession"
    # Matches straight quotes (single/double) and curly quotes.
    quoted = re.search(r'["‘’“”\'](.+?)["‘’“”\']', query)
    if quoted:
        return quoted.group(1).strip()

    # Strip leading question phrases: "what is the theme of", "tell me about",
    # "who directed", "when was", "what year was", "where can I watch", etc.
    stripped = re.sub(
        r"^\s*(?:what\s+(?:is|are|was|were)\s+(?:the\s+)?"
        r"(?:(?:theme|plot|story|genre|cast|director|rating|year|overview|about|runtime|tagline|budget)\s+of\s+|about\s+)?|"
        r"who\s+(?:directed|starred\s+in|wrote|made|produced)\s+|"
        r"when\s+(?:was|is|did|released|does|will)\s+(?:the\s+)?(?:film\s+|movie\s+)?|"
        r"where\s+can\s+i\s+(?:watch|stream|find|see)\s+|"
        r"tell\s+me\s+about\s+(?:the\s+(?:film\s+|movie\s+))?|"
        r"(?:the\s+)?(?:film|movie)\s+|"
        r"recommend\s+(?:me\s+)?(?:a\s+|some\s+)?|"
        r"suggest\s+(?:me\s+)?(?:a\s+|some\s+)?|"
        r"find\s+(?:me\s+)?(?:a\s+|some\s+)?)",
        "",
        query,
        flags=re.IGNORECASE,
    ).strip().rstrip("?.!")
    # Strip trailing verb phrases that are part of the question phrasing, not
    # the title: "Obsession released" → "Obsession", "Inception made" → "Inception".
    stripped = re.sub(
        r"\s+(?:released|made|directed|produced|written|filmed|set|come\s+out|came\s+out)$",
        "",
        stripped,
        flags=re.IGNORECASE,
    ).strip()
    # Strip a trailing 4-digit year (the title payload field has no year).
    stripped = re.sub(r"\s+(?:19|20)\d{2}$", "", stripped).strip()
    # Only trust the result when it looks like a title: non-empty and not a
    # common pronoun / filler word (which indicate we didn't strip enough).
    if stripped and not re.match(
        r"^(?:it|that|this|the|a|an|something|anything|everything|nothing)\b",
        stripped,
        re.IGNORECASE,
    ):
        # Corpus titles are TMDB-cased (e.g. "Obsession"); user queries are
        # often lowercase ("when released obsession?"). Exact-title Qdrant
        # lookup is case-sensitive, so title-case unless the query already
        # supplied mixed case (respect an explicitly-quoted/typed title).
        if stripped == stripped.lower() or stripped == stripped.upper():
            return stripped.title()
        return stripped
    return None


@lru_cache(maxsize=256)
def _extract_title_via_llm(query: str) -> str | None:
    """LLM fallback for title extraction — only called on regex miss.

    Cheap model, one short call, cached per query. Any API error or a
    "NONE" response returns None (caller treats as no-collision-lookup).
    """
    from openai import OpenAI

    prompt = _EXTRACT_PROMPT_TEMPLATE.format(query=query)
    try:
        response = OpenAI().chat.completions.create(
            model=_EXTRACT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=20,
            temperature=0.0,
        )
        title = (response.choices[0].message.content or "").strip()
    except Exception:  # noqa: BLE001 — API failure must not block the run
        return None
    return title if title and title.upper() != "NONE" else None


# ---------------------------------------------------------------------------
# Year-pinned query check
# ---------------------------------------------------------------------------

_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")


def _query_pins_year(query: str) -> bool:
    """Return True if the query already contains a 4-digit year.

    A year-pinned query (e.g. "Obsession 2026") resolves cleanly via normal
    retrieval (0005 AC-9) and must not trigger a disambiguation turn.
    """
    return bool(_YEAR_RE.search(query))


# ---------------------------------------------------------------------------
# Point-id scheme (mirrors ingestion/retrieval convention)
# ---------------------------------------------------------------------------


def _tmdb_point_id(tmdb_id: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, str(tmdb_id)))


# ---------------------------------------------------------------------------
# Collision detection (pre-graph gate)
# ---------------------------------------------------------------------------


def detect_title_collision(
    query: str,
    *,
    settings: object,  # RetrievalSettings — imported lazily to avoid circular dep
) -> TitleCollision | None:
    """Return a ``TitleCollision`` when the query names a same-title collision.

    Steps:
    1. If the query contains a 4-digit year → return None (already pinned).
    2. Extract a title from the query → return None if not extractable.
    3. ``find_by_exact_title(title)`` → return None if < 2 hits.
    4. Return a populated ``TitleCollision``.

    The caller is responsible for only invoking this on inform-shaped queries
    (AC-8 scope guard).  This function itself has no intent classifier — it
    simply checks: is a title extractable? are there ≥2 exact-title matches?
    """
    # Step 1: year already pinned → normal retrieval handles it
    if _query_pins_year(query):
        return None

    # Step 2: extract title — regex fast path first (free, no LLM, covers the
    # common phrasings). Falls back to a cheap LLM call only when regex can't
    # parse the query at all (~5% of queries), or when regex extracted a
    # title that hits zero corpus results (garbage extraction — same symptom
    # as no-title, just a validity check away).
    title = extract_title_from_query(query)
    hits = find_by_exact_title(title, settings=settings) if title else []  # type: ignore[arg-type]

    if not hits:
        llm_title = _extract_title_via_llm(query)
        if not llm_title:
            return None
        title = llm_title
        hits = find_by_exact_title(title, settings=settings)  # type: ignore[arg-type]

    # Step 3: need ≥2 exact-title matches to count as a collision
    if len(hits) < 2:
        return None

    # Step 4: collision confirmed
    candidates = [
        CollisionCandidate(tmdb_id=h.tmdb_id, year=h.year)
        for h in hits
        if h.tmdb_id and h.year
    ]
    if len(candidates) < 2:
        return None

    return TitleCollision(title=title, candidates=candidates)


# ---------------------------------------------------------------------------
# Templated clarification question
# ---------------------------------------------------------------------------


def build_collision_question(collision: TitleCollision) -> str:
    """Build the deterministic, templated disambiguation question.

    Matches the observed real-transcript wording (years ascending).
    """
    sorted_years = sorted(c.year for c in collision.candidates)
    year_str = ", ".join(str(y) for y in sorted_years)
    n = len(collision.candidates)
    return (
        f"There are {n} films called {collision.title}: "
        f"from {year_str} — which one did you mean?"
    )


# ---------------------------------------------------------------------------
# Year-reference resolution (pure, deterministic)
# ---------------------------------------------------------------------------

_ORDINALS: dict[str, int] = {
    "first": 1, "1st": 1,
    "second": 2, "2nd": 2,
    "third": 3, "3rd": 3,
    "fourth": 4, "4th": 4,
    "fifth": 5, "5th": 5,
}


def resolve_year_reference(
    answer: str,
    candidates: list[CollisionCandidate],
) -> int | None:
    """Map a free-text follow-up to a single tmdb_id, or None if unresolvable.

    Handles:
    - **Exact year**: "the 1976 one", "1976", "the one from 1976" → 1976 film.
    - **Superlative / relative**: "the newest one" / "latest" → max year;
      "the oldest" / "original" / "first" / "earliest" → min year.
    - **Ordinal**: "the second one" → 2nd candidate by ascending year.
    - **Nearest year** (fuzzy): a year not in the offered set maps to the
      nearest candidate within MAX_YEAR_DISTANCE (ties → newer). Beyond that
      tolerance → None. The real-transcript case is "the 2025 one" against
      {1943,1976,2015,2026} → 2026 (distance 1).
    - **Unresolvable** → None.

    Returns the tmdb_id of the resolved film, or None.
    """
    if not candidates:
        return None

    text = answer.lower().strip()
    sorted_cands = sorted(candidates, key=lambda c: c.year)

    # -----------------------------------------------------------------------
    # Superlatives — check before ordinal and year so "first" is unambiguous
    # -----------------------------------------------------------------------
    if any(tok in text for tok in ("newest", "latest", "most recent", "recent")):
        return sorted_cands[-1].tmdb_id
    if any(tok in text for tok in ("oldest", "earliest")):
        return sorted_cands[0].tmdb_id
    if any(tok in text for tok in ("original",)):
        return sorted_cands[0].tmdb_id

    # -----------------------------------------------------------------------
    # Ordinals (before "first" is caught as a year keyword)
    # -----------------------------------------------------------------------
    for word, rank in _ORDINALS.items():
        if word in text:
            # "first" is also a superlative handled above; by the time we
            # reach here the superlative pass already returned for "oldest/
            # original/earliest", so "first" here means rank-1 ordinal.
            if rank <= len(sorted_cands):
                return sorted_cands[rank - 1].tmdb_id
            return None  # out-of-bounds ordinal → unresolvable

    # -----------------------------------------------------------------------
    # Explicit 4-digit year
    # -----------------------------------------------------------------------
    year_match = _YEAR_RE.search(text)
    if year_match:
        typed_year = int(year_match.group())

        # Exact match first
        for cand in sorted_cands:
            if cand.year == typed_year:
                return cand.tmdb_id

        # Nearest match within tolerance
        best: CollisionCandidate | None = None
        best_dist = MAX_YEAR_DISTANCE + 1  # sentinel: one beyond tolerance

        for cand in sorted_cands:
            dist = abs(cand.year - typed_year)
            if dist < best_dist:
                best_dist = dist
                best = cand
            elif dist == best_dist and best is not None and cand.year > best.year:
                # Tie-break: prefer newer film
                best = cand

        if best is not None and best_dist <= MAX_YEAR_DISTANCE:
            return best.tmdb_id

        # Out-of-tolerance → None (AC-7 fallback in the caller)
        return None

    # -----------------------------------------------------------------------
    # Unresolvable
    # -----------------------------------------------------------------------
    return None


# ---------------------------------------------------------------------------
# Single-film fetch by tmdb_id (for seeding resolved_inform_tmdb_id)
# ---------------------------------------------------------------------------


def fetch_film_by_tmdb_id(
    tmdb_id: int,
    *,
    settings: object,  # RetrievalSettings
) -> dict | None:
    """Fetch a single film's payload from Qdrant by its derived point id.

    Returns a MovieHit serialised as a dict (compatible with ``rag_hits``),
    or None if the id is not found in the corpus.
    """
    from retrieval.client import get_qdrant_client

    point_id = _tmdb_point_id(tmdb_id)
    client = get_qdrant_client()

    try:
        records = client.retrieve(
            collection_name=settings.ingestion().movies_collection,  # type: ignore[attr-defined]
            ids=[point_id],
            with_payload=True,
            with_vectors=False,
        )
    except Exception:  # noqa: BLE001 — don't let a lookup error crash the run
        return None

    if not records:
        return None

    p = records[0].payload or {}
    return {
        "tmdb_id": p.get("tmdb_id", tmdb_id),
        "title": p.get("title", ""),
        "year": p.get("year", 0),
        "overview": p.get("overview", ""),
        "genres": p.get("genres", []),
        "vote_average": p.get("vote_average", 0.0),
        "score": 0.0,
    }
