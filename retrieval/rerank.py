"""Cross-encoder reranking via sentence-transformers."""

from __future__ import annotations

import functools
import logging
from typing import Union

from retrieval.models import MovieHit, ReviewHit

_logger = logging.getLogger(__name__)

Hit = Union[MovieHit, ReviewHit]

# STS-B cross-encoder: trained on short sentence-pair similarity — same text
# shape as title+overview — rather than long MS-MARCO web query→passage pairs.
_DEFAULT_MODEL = "cross-encoder/stsb-distilroberta-base"


@functools.lru_cache(maxsize=4)
def _load_cross_encoder(model: str):  # type: ignore[return]
    from sentence_transformers import CrossEncoder  # lazy — only when rerank=True

    return CrossEncoder(model)


def cross_encode_rerank(
    query: str,
    hits: list[Hit],
    *,
    model: str = _DEFAULT_MODEL,
) -> list[Hit]:
    """Reorder hits by cross-encoder score (higher = more relevant).

    Accepts either MovieHit or ReviewHit. Works on mixed lists only if
    the caller guarantees hit.chunk_text / hit.overview as the passage text.
    """
    if not hits:
        return hits

    encoder = _load_cross_encoder(model)

    def _text(hit: Hit) -> str:
        if isinstance(hit, ReviewHit):
            return hit.chunk_text
        return f"{hit.title} {hit.overview}"

    pairs = [(query, _text(h)) for h in hits]
    scores: list[float] = encoder.predict(pairs).tolist()

    ranked = sorted(zip(hits, scores), key=lambda x: x[1], reverse=True)
    _logger.debug('{"step":"rerank","model":"%s","hits":%d}', model, len(hits))
    return [h for h, _ in ranked]
