"""Shared ingestion settings — single source of truth for embedder, chunk params, and collection names."""

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    embedder: Literal["openai-3-small", "openai-3-large", "minilm"] = "openai-3-small"
    chunk_max_tokens: int = 300
    chunk_overlap_tokens: int = 50

    @property
    def embedder_dim(self) -> int:
        return {"openai-3-small": 1536, "openai-3-large": 3072, "minilm": 384}[self.embedder]

    @property
    def variant_suffix(self) -> str:
        if self.embedder == "openai-3-small":
            return "3small"
        if self.embedder == "openai-3-large":
            return "3large"
        return f"minilm_c{self.chunk_max_tokens}o{self.chunk_overlap_tokens}"

    def _is_default_variant(self) -> bool:
        return (
            self.embedder == "openai-3-small"
            and self.chunk_max_tokens == 300
            and self.chunk_overlap_tokens == 50
        )

    @property
    def movies_collection(self) -> str:
        if self._is_default_variant():
            return "tmdb_movies"
        return f"tmdb_movies__{self.variant_suffix}"

    @property
    def reviews_collection(self) -> str:
        if self._is_default_variant():
            return "tmdb_reviews"
        return f"tmdb_reviews__{self.variant_suffix}"
