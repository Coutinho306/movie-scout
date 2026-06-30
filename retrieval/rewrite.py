"""LLM-based query rewriting for retrieval optimization."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from openai import OpenAI

_PROMPT_PATH = Path(__file__).parent / "prompts" / "rewrite.md"
_PROMPT_TEMPLATE = _PROMPT_PATH.read_text()

_DEFAULT_MODEL = "gpt-4o-mini"

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


@lru_cache(maxsize=256)
def rewrite_query(query: str, *, model: str = _DEFAULT_MODEL) -> str:
    """Return a retrieval-optimized rewrite of query.

    Cached per (query, model) so repeated calls in a session are free.
    Falls back to the original query on API error.
    """
    model = model or os.environ.get("MODEL_ORCHESTRATOR", _DEFAULT_MODEL)
    prompt = _PROMPT_TEMPLATE.format(query=query)
    try:
        response = _get_client().chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=80,
            temperature=0.0,
        )
        rewritten = (response.choices[0].message.content or "").strip()
        return rewritten if rewritten else query
    except Exception:
        return query
