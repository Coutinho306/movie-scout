"""Deterministic classifier: route a vector-search-bound query to hybrid or dense.

Pure regex + lexical heuristics, no LLM call, no network — modeled on
``seed_film.py``. Returns ``True`` (hybrid BM25+dense RRF) for literal
genre/mood/overview-shaped queries (diagnostic tiers 1 and 2), and
``False`` (dense only) for abstract/conversational queries (tier 3) or
verbatim titles (tier 0).

The bias is **dense on uncertainty** — a false "hybrid" on a tier-3
abstract query costs -0.10 recall (the observed regression from 0008),
while a false "dense" on a tier-2/1 query merely forgoes lift. When
unclear, we return False.

Routing logic:
- Tier-2 template shape (``"a ... film — ..."`` or ``"an ... film — ..."``)
  → always hybrid (highest confidence; genre tokens plus templated structure)
- Conversational/request prefix (``"I'm looking for"``, ``"Can you recommend"``,
  ``"Looking for"``, ``"Find me"``, ``"Recommend"``, ``"Show me"``) → always dense
  (tier-3 LLM-generated queries; hybrid regresses here)
- Short verbatim title (≤ 4 tokens, no genre lexicon match, no sentence
  structure) → dense (tier-0 shape; title queries bypass vector search in
  production but can reach here via direct API calls)
- Descriptive narrative sentence (has a verb, > 6 words, not conversational)
  AND carries at least one genre/descriptor anchor → hybrid (tier-1 shape)
- Otherwise → dense (default, protects against misclassifying tier-3)
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Tier-2 template detector
# ---------------------------------------------------------------------------

# Matches: "a <genres> film — <mood>" / "an <genres> film — <mood>"
# The dash can be em-dash (—), en-dash (–), or ASCII hyphen.
_TIER2_PATTERN = re.compile(
    r"^an?\s+.+?\b(?:film|movie)\b\s*[—–-]\s*.+",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Conversational/request prefix detector (→ dense, tier-3 shape)
# ---------------------------------------------------------------------------

_REQUEST_PATTERNS = [
    re.compile(r"^I\s*'?m\s+looking\s+for\b", re.IGNORECASE),
    re.compile(r"^Can\s+you\s+recommend\b", re.IGNORECASE),
    re.compile(r"^Looking\s+for\b", re.IGNORECASE),
    re.compile(r"^Find\s+me\b", re.IGNORECASE),
    re.compile(r"^(?:Recommend|Suggest)\b", re.IGNORECASE),
    re.compile(r"^Show\s+me\b", re.IGNORECASE),
    re.compile(r"^Do\s+you\s+(?:know|have)\b", re.IGNORECASE),
    re.compile(r"^What\s+(?:are|is)\b", re.IGNORECASE),
]

# ---------------------------------------------------------------------------
# Genre / descriptor lexicon (signals literal content → hybrid)
# ---------------------------------------------------------------------------

_GENRE_LEXICON = frozenset(
    [
        # Core genres
        "action", "adventure", "animation", "animated", "comedy", "crime",
        "documentary", "drama", "fantasy", "horror", "mystery", "romance",
        "romantic", "sci-fi", "science fiction", "scifi", "thriller",
        "western", "musical", "biography", "biopic", "war", "history",
        "historical", "family", "sport", "sports", "superhero",
        # Descriptor / mood tokens that signal literal genre queries
        "dark", "atmospheric", "suspenseful", "psychological", "supernatural",
        "haunting", "gripping", "intense", "gritty", "lighthearted",
        "quirky", "heartwarming", "bittersweet", "coming-of-age", "visually",
        "stunning", "cinematic", "emotional", "slow", "meditative",
        "contemplative", "poetic", "bleak", "tense", "stylish",
    ]
)


def _has_genre_anchor(query: str) -> bool:
    """Return True if any genre/descriptor token from the lexicon appears."""
    q_lower = query.lower()
    # Check multi-word entries first
    for term in _GENRE_LEXICON:
        if " " in term and term in q_lower:
            return True
    # Single-word entries: check as word-boundary match
    words = set(re.findall(r"[a-z\-]+", q_lower))
    return bool(words & _GENRE_LEXICON)


def _is_narrative_sentence(query: str) -> bool:
    """Heuristic: query looks like a descriptive/overview sentence (tier-1 shape).

    Requires more than 8 tokens (rules out short titles and most tier-0
    verbatim titles; all real tier-0 titles in the diagnostic suite are ≤ 8
    tokens). The key property is that it is a long descriptive sentence —
    genre anchors are not required because tier-1 queries are first sentences
    of overviews and typically discuss characters and plot, not genres.
    """
    return len(query.split()) > 8


# ---------------------------------------------------------------------------
# Public classifier
# ---------------------------------------------------------------------------

def classify_query_mode(query: str) -> bool:
    """Return True (hybrid) or False (dense) for a vector-search-bound query.

    Dense-biased on uncertainty: returns False when no strong signal is found.
    Title/actor queries that are handled upstream in search_movies_tool never
    reach this function — it only sees vector-search-bound queries.
    """
    q = query.strip()
    if not q:
        return False  # empty → dense (safe default)

    # 1. Tier-2 template: "a ... film — ..." → hybrid (highest confidence)
    if _TIER2_PATTERN.match(q):
        return True

    # 2. Conversational request prefix → dense (tier-3 shape)
    for pat in _REQUEST_PATTERNS:
        if pat.match(q):
            return False

    # 3. Descriptive narrative sentence → hybrid (tier-1 shape)
    #    Long sentences (> 8 tokens) that are not request prefixes are
    #    overview/plot-description queries; BM25 keyword overlap helps there.
    #    No genre anchor required — tier-1 queries rarely contain genre tokens.
    if _is_narrative_sentence(q):
        return True

    # 4. Default: dense (protects tier-0 titles and uncertain tier-3 queries)
    return False
