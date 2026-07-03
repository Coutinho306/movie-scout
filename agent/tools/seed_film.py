"""Deterministic extraction of a named seed film from a free-text query.

Pure regex, no LLM call: covers the seed-film phrasings observed in manual
testing ("a film like X", "same theme as X", "similar to X", "movies like X",
"in the style of X"). Falls through to None on no match — callers must treat
that as "no seed named, use plain text search" rather than an error.
"""

from __future__ import annotations

import re

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
