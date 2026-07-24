"""Deterministic input classifier: detect prompt-injection query shapes.

Pure regex/lexical rules only — no LLM, no network.  Mirrors the style of
``agent/tools/query_mode.py``: module-level compiled patterns, one public pure
function, safe-default-on-uncertainty (returns ``"ok"`` when nothing matches).

The classifier is deliberately narrow — it catches the obvious/demoable shapes
(API-key asks, ignore-instructions phrasing, reveal-system-prompt phrasing) and
does NOT attempt off-domain enumeration.  The output score-floor gate is the
backstop for anything that slips past (e.g., paraphrased injections, off-topic
queries).
"""

from __future__ import annotations

import re
from typing import Literal

# ---------------------------------------------------------------------------
# Injection pattern set
# ---------------------------------------------------------------------------

_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    # API / secret key asks
    re.compile(r"\bapi\s*[-–]?\s*key\b", re.IGNORECASE),
    re.compile(r"\bopenai\s+(?:api\s*)?key\b", re.IGNORECASE),
    re.compile(r"\bsecret\s+key\b", re.IGNORECASE),

    # "Ignore / disregard / forget (all|your|previous|prior) (prompt|instructions|commands|rules)"
    re.compile(
        r"\b(?:ignore|disregard|forget)\b.{0,40}\b(?:previous|prior|all|your)?\b.{0,20}"
        r"\b(?:prompt|instruction|command|rule)s?\b",
        re.IGNORECASE | re.DOTALL,
    ),

    # "system prompt" standalone (reveal, show, etc.)
    re.compile(r"\bsystem\s+prompt\b", re.IGNORECASE),

    # "reveal/print/show/dump/leak (me) (your) (prompt|instructions|system|rules|commands)"
    re.compile(
        r"\b(?:reveal|print|dump|leak)\b.{0,40}\b(?:prompt|instruction|system|rule|command)s?\b",
        re.IGNORECASE | re.DOTALL,
    ),
    # "show me your (prompt|instructions|system prompt)" — "show" alone triggers innocuous
    # "show me good X" queries, so scope to: show ... your ... (prompt|instruction|...)
    re.compile(
        r"\bshow\b.{0,20}\byour\b.{0,20}\b(?:prompt|instruction|system|rule|command)s?\b",
        re.IGNORECASE | re.DOTALL,
    ),
]

# ---------------------------------------------------------------------------
# Public classifier
# ---------------------------------------------------------------------------

QueryScope = Literal["ok", "injection"]


def classify_query_scope(query: str) -> QueryScope:
    """Return ``"injection"`` if the query matches a known injection pattern.

    Safe-default: returns ``"ok"`` when the query is empty or no pattern
    matches.  Wrapping the call in a ``try/except`` is recommended for callers
    on the hot path (see ``agent/main.py::run``).

    Parameters
    ----------
    query:
        Raw user query string, any casing/whitespace.

    Returns
    -------
    ``"injection"`` when a prompt-injection shape is detected;
    ``"ok"`` otherwise (including on empty input).
    """
    q = query.strip()
    if not q:
        return "ok"

    for pat in _INJECTION_PATTERNS:
        if pat.search(q):
            return "injection"

    return "ok"
