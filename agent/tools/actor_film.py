"""Deterministic extraction of a named actor from a free-text query.

Pure regex, no LLM call: covers high-precision actor-query phrasings observed in
manual testing. Biased for **precision over recall** — a missed extraction falls
through to dense semantic search (the status quo, no regression), whereas a false
match would route a non-actor query to an empty cast filter.

Covered phrasings (exact trigger set):
  - "films with Ryan Gosling"
  - "movies with Ryan Gosling"
  - "films with the actor Ryan Gosling"
  - "movies with the actor Ryan Gosling"
  - "starring Ryan Gosling"
  - "with the actor Ryan Gosling"
  - "the actor Ryan Gosling" (start of query)

Deliberately NOT covered (graceful degradation to dense search):
  - Leading-name forms: "Ryan Gosling movies", "Keanu Reeves films"
  - Bare "Ryan Gosling" with no trigger phrase
  - Multi-word/complex phrasings not in the tested trigger set

The covered set matches what manual testing showed as the actual user phrasings
for cast-filtered queries. See seed_film.py for the parallel pattern this mirrors.
"""

from __future__ import annotations

import re

# Stop capture at a clause boundary: comma, period, semicolon, or a small set
# of trailing connectors. Mirrors seed_film.py's _STOP pattern.
_STOP = r"(?=[,.;?]|\s+(?:featuring|directed by|that|which|from|in|and)\b|$)"

# Capture a proper-noun actor name: one or more Title-Case words (or initials).
# Stops at the _STOP boundary defined above.
_NAME = rf"([A-Z][a-zA-Z'-]+(?:\s+[A-Z][a-zA-Z'-]+)+){_STOP}"

_ACTOR_PATTERNS: list[re.Pattern[str]] = [
    # "films/movies with (the actor) X"
    # Negative lookahead excludes seed-film phrasings such as "with the same
    # theme/vibe as" which share the "films? with" prefix.
    re.compile(
        rf"(?:films?|movies?)\s+with\s+(?!the\s+same\b)(?:the\s+actor\s+)?{_NAME}",
        re.IGNORECASE,
    ),
    # "starring X"
    re.compile(
        rf"starring\s+{_NAME}",
        re.IGNORECASE,
    ),
    # "(with) the actor X"  — also catches "with the actor X" standalone
    re.compile(
        rf"(?:with\s+)?the\s+actor\s+{_NAME}",
        re.IGNORECASE,
    ),
]


def extract_actor_name(query: str) -> str | None:
    """Return the actor name from a high-precision actor-query phrasing, or None.

    Returns None for phrasings not in the covered trigger set — callers must
    treat that as "no actor detected, use dense search" rather than an error.
    """
    for pattern in _ACTOR_PATTERNS:
        match = pattern.search(query)
        if match:
            name = match.group(1).strip().rstrip("?.!,")
            if name:
                return name
    return None
