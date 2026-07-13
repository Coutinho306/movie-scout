"""Seed-film extraction and seed-intent classification.

Two extraction strategies, cost-ordered:
1. ``extract_seed_title`` — deterministic regex fast-path (EN phrasings,
   free, no LLM call). Falls through to None on no pattern match.
2. ``is_seed_intent`` — cheap LLM fallback (``gpt-4o-mini``) scoped to the
   seed-candidate branch only (i.e. called only when the regex path returned
   None). Classifies multilingual seed queries and extracts the named film.
   Zero calls on non-seed traffic — the cost property the project prizes.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

# Capture up to the next clause boundary (comma/period/semicolon or a small
# set of trailing connector words), NOT to end-of-string — the query-rewrite
# node (retrieval/rewrite.py) expands short seed-film queries into long
# descriptive sentences ("a film like Arrival" -> "a thought-provoking sci-fi
# film similar to Arrival, featuring complex narratives, ... preferably
# directed by Denis Villeneuve..."), and a greedy end-of-string capture would
# swallow all of that trailing description as part of the "title".
_STOP = r"(?=[,.;]|\s+(?:featuring|with|preferably|directed by|starring|that|which)\b|$)"

_SEED_PATTERNS = [
    re.compile(rf"(?:a\s+)?(?:[\w-]+\s+){{0,4}}(?:film|movie)s?\s+like\s+(.+?){_STOP}", re.IGNORECASE),
    re.compile(
        rf"(?:a\s+)?(?:[\w-]+\s+){{0,4}}(?:film|movie)\s+(?:with\s+the\s+)?same\s+(?:theme|vibe|energy)s?\s+as\s+(.+?){_STOP}",
        re.IGNORECASE,
    ),
    re.compile(rf"(?:a\s+)?(?:[\w-]+\s+){{0,4}}(?:film|movie)\s+similar\s+to\s+(.+?){_STOP}", re.IGNORECASE),
    re.compile(rf"similar\s+to\s+(.+?){_STOP}", re.IGNORECASE),
    re.compile(rf"in\s+the\s+style\s+of\s+(.+?){_STOP}", re.IGNORECASE),
]


def extract_seed_title(query: str) -> str | None:
    """Return the named seed film's title substring, or None if no pattern matches."""
    for pattern in _SEED_PATTERNS:
        match = pattern.search(query)
        if match:
            title = match.group(1).strip().rstrip("?.!,")
            if title:
                return title
    return None


# ---------------------------------------------------------------------------
# LLM-based seed-intent classifier (multilingual, cheap-model fallback)
# ---------------------------------------------------------------------------

_SEED_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "extract_seed_title.md"
_SEED_PROMPT_TEMPLATE = _SEED_PROMPT_PATH.read_text()
_SEED_EXTRACT_MODEL = "gpt-4o-mini"


@lru_cache(maxsize=256)
def _extract_seed_title_via_llm(query: str) -> str | None:
    """Ask a cheap LLM to extract the seed-film title from a multilingual query.

    Reuses the same extraction-prompt pattern as ``disambiguation._extract_title_via_llm``
    but targets seed-intent ("films like X") rather than inform-intent ("tell me about X").

    Returns the bare title string if the query is seed-shaped, or None if the LLM
    says NONE (not a seed query) or if the API call fails. Cached per query.
    """
    from openai import OpenAI

    prompt = _SEED_PROMPT_TEMPLATE.format(query=query)
    try:
        response = OpenAI().chat.completions.create(
            model=_SEED_EXTRACT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=20,
            temperature=0.0,
        )
        title = (response.choices[0].message.content or "").strip()
    except Exception:  # noqa: BLE001 — API failure must not crash the gate
        return None
    return title if title and title.upper() != "NONE" else None


def is_seed_intent(query: str) -> bool:
    """Return True if the query is a seed-film / "films like X" request.

    Cost contract: this function is scoped to the *already-narrowed* seed-candidate
    branch inside ``detect_franchise_ambiguity`` — it is called only after
    ``extract_seed_title`` returned None (i.e. the EN regex fast-path missed).
    Generic recommend queries, inform queries, and actor queries that never reach
    ``detect_franchise_ambiguity`` at all will never trigger this function.

    Internally calls ``_extract_seed_title_via_llm`` and returns True iff the LLM
    extracts a non-None title (the LLM both classifies and extracts in one call).
    """
    return _extract_seed_title_via_llm(query) is not None
