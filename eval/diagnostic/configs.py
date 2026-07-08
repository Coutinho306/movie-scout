"""Named retrieval config matrix for the diagnostic suite.

Each ``DiagnosticConfig`` describes one retrieval path to evaluate:
- ``settings_kwargs`` are passed to ``RetrievalSettings(**settings_kwargs)``.
- ``hyde_blend_alpha`` (optional): if set, the runner sets ``HYDE_BLEND_ALPHA``
  in the environment before calling ``search_movies``.  If None and
  ``query_rewrite=True``, ``HYDE_BLEND_ALPHA`` is unset (pure HyDE).
- ``rerank=True``: the runner fetches ``prefetch_k`` results then applies
  ``cross_encode_rerank`` down to ``top_k``.  Do NOT rely on
  ``settings.rerank`` — it is a dead flag in ``search_movies``.

Adding a new config requires only adding an entry here; no other module changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DiagnosticConfig:
    """One named retrieval configuration in the diagnostic matrix."""

    name: str
    # kwargs forwarded to RetrievalSettings(); top_k/hybrid/query_rewrite etc.
    settings_kwargs: dict = field(default_factory=dict)
    # Set HYDE_BLEND_ALPHA env var to this value when calling search_movies.
    # None and query_rewrite=True => unset env var (pure HyDE).
    # Ignored when query_rewrite=False.
    hyde_blend_alpha: float | None = None
    # Whether to apply cross_encode_rerank after fetching prefetch_k results.
    rerank: bool = False
    # How many results to fetch before reranking (ignored when rerank=False).
    prefetch_k: int = 50


CONFIGS: list[DiagnosticConfig] = [
    DiagnosticConfig(
        name="baseline_dense",
        settings_kwargs={"hybrid": False, "query_rewrite": False},
    ),
    DiagnosticConfig(
        name="hyde_blended",
        settings_kwargs={"hybrid": False, "query_rewrite": True},
        hyde_blend_alpha=0.5,
    ),
    DiagnosticConfig(
        name="hyde_pure",
        settings_kwargs={"hybrid": False, "query_rewrite": True},
        hyde_blend_alpha=None,  # leave HYDE_BLEND_ALPHA unset => pure HyDE
    ),
    DiagnosticConfig(
        name="rerank_widened",
        settings_kwargs={"hybrid": False, "query_rewrite": False},
        rerank=True,
        prefetch_k=50,
    ),
    DiagnosticConfig(
        name="hyde_rerank",
        settings_kwargs={"hybrid": False, "query_rewrite": True},
        hyde_blend_alpha=0.5,
        rerank=True,
        prefetch_k=50,
    ),
    DiagnosticConfig(
        name="hybrid_bm25",
        settings_kwargs={"hybrid": True, "query_rewrite": False},
    ),
]
