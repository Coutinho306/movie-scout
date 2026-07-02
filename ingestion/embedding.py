"""Embedder protocol and implementations (OpenAI 3-small/3-large, local MiniLM)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import openai

if TYPE_CHECKING:
    from ingestion.config import Settings

_logger = logging.getLogger(__name__)


@runtime_checkable
class Embedder(Protocol):
    """embed_texts = document path (ingest); embed_single = query path
    (may prepend a model-specific query instruction, e.g. bge)."""

    model: str
    dim: int

    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...
    def embed_single(self, text: str) -> list[float]: ...


class OpenAIEmbedder:
    def __init__(self, model: str, dim: int) -> None:
        self.model = model
        self.dim = dim
        self._client: openai.OpenAI | None = None

    def _get_client(self) -> openai.OpenAI:
        if self._client is None:
            self._client = openai.OpenAI()
        return self._client

    def embed_texts(self, texts: list[str], *, batch_size: int = 100) -> list[list[float]]:
        results: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self._get_client().embeddings.create(input=batch, model=self.model)
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

    def embed_single(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]


class LocalEmbedder:
    def __init__(self, model: str = "all-MiniLM-L6-v2", query_prefix: str = "") -> None:
        self.model = model
        self.dim = 384
        # Some models (e.g. bge) expect an instruction prepended to *queries* only;
        # documents are embedded as-is via embed_texts.
        self.query_prefix = query_prefix
        self._st = None

    def _get_st(self):  # type: ignore[return]
        if self._st is None:
            from sentence_transformers import SentenceTransformer  # lazy import

            self._st = SentenceTransformer(self.model)
        return self._st

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vecs = self._get_st().encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return [v.tolist() for v in vecs]

    def embed_single(self, text: str) -> list[float]:
        return self.embed_texts([self.query_prefix + text])[0]


_OPENAI_MODELS = {
    "openai-3-small": ("text-embedding-3-small", 1536),
    "openai-3-large": ("text-embedding-3-large", 3072),
}

_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def get_embedder(settings: Settings) -> Embedder:
    if settings.embedder in _OPENAI_MODELS:
        model_name, dim = _OPENAI_MODELS[settings.embedder]
        return OpenAIEmbedder(model=model_name, dim=dim)
    if settings.embedder == "bge-small":
        return LocalEmbedder(model="BAAI/bge-small-en-v1.5", query_prefix=_BGE_QUERY_PREFIX)
    return LocalEmbedder()
