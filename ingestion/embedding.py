"""OpenAI embedding helpers (text-embedding-3-small, 1536 dims)."""

import logging

import openai

_client: openai.OpenAI | None = None
_logger = logging.getLogger(__name__)


def _get_client() -> openai.OpenAI:
    global _client
    if _client is None:
        _client = openai.OpenAI()
    return _client


def embed_texts(
    texts: list[str],
    *,
    model: str = "text-embedding-3-small",
    batch_size: int = 100,
) -> list[list[float]]:
    results: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = _get_client().embeddings.create(input=batch, model=model)
        tokens = response.usage.total_tokens
        cost = tokens / 1000 * 0.00002
        _logger.info(
            '{"step":"embed","chunks":%d,"tokens":%d,"cost_usd":%.6f}',
            len(batch),
            tokens,
            cost,
        )
        results.extend(item.embedding for item in response.data)
    return results


def embed_single(text: str) -> list[float]:
    return embed_texts([text])[0]
