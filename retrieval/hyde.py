"""HyDE (Hypothetical Document Embeddings) query expansion for retrieval.

Instead of embedding the abstract user query directly, asks an LLM to write
a concrete TMDB-style movie overview that would satisfy the query, then embeds
that hypothetical document.  The concrete vocabulary of the hypothetical doc
lives in the same neighbourhood as the stored movie embeddings, bridging the
abstract-query ↔ concrete-document gap that collapses recall at full corpus
scale.

This is deliberately separate from retrieval.rewrite (which stays abstract and
optimises synonym coverage for keyword-style retrieval) — the two prompts solve
different problems.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from openai import OpenAI

if TYPE_CHECKING:
    from ingestion.embedding import Embedder

_PROMPT_PATH = Path(__file__).parent / "prompts" / "hyde.md"
_PROMPT_TEMPLATE = _PROMPT_PATH.read_text()

_DEFAULT_MODEL = "gpt-4o-mini"

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


@lru_cache(maxsize=256)
def generate_hyde_text(query: str, *, model: str = _DEFAULT_MODEL) -> str:
    """Return a hypothetical concrete movie overview that satisfies query.

    Cached per (query, model).  Falls back to the original query on any
    LLM error so callers can treat the return value as always usable.
    """
    model = model or os.environ.get("MODEL_ORCHESTRATOR", _DEFAULT_MODEL)
    prompt = _PROMPT_TEMPLATE.format(query=query)
    try:
        response = _get_client().chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.3,
        )
        text = (response.choices[0].message.content or "").strip()
        return text if text else query
    except Exception:
        return query


def hyde_embed(
    query: str,
    embedder: "Embedder",
    *,
    model: str = _DEFAULT_MODEL,
    blend_alpha: float | None = None,
) -> list[float]:
    """Return an embedding for query using HyDE.

    Parameters
    ----------
    query:
        The raw user query.
    embedder:
        The same embedder used to index the corpus (must match vector space).
    model:
        LLM used to generate the hypothetical document.
    blend_alpha:
        If None, return the pure HyDE vector (embed the hypothetical doc only).
        If a float in [0, 1], return ``alpha * query_vec + (1-alpha) * hyde_vec``
        so the raw query anchors the direction.  0.5 gives equal weight.
    """
    hyde_text = generate_hyde_text(query, model=model)
    hyde_vec = embedder.embed_single(hyde_text)

    if blend_alpha is None:
        return hyde_vec

    query_vec = embedder.embed_single(query)
    blended = [
        blend_alpha * q + (1.0 - blend_alpha) * h
        for q, h in zip(query_vec, hyde_vec)
    ]
    return blended
