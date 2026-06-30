"""Retrieval-time settings — knobs that don't affect ingestion."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict

from ingestion.config import Settings as IngestionSettings


class RetrievalSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    top_k: int = 10
    hybrid: bool = False
    rerank: bool = False
    query_rewrite: bool = False
    score_threshold: float | None = None
    taste_alpha: float = 0.5  # weight on retrieval_score in blended rank

    def ingestion(self) -> IngestionSettings:
        """Return an IngestionSettings from the same .env — shared embedder + collection names."""
        return IngestionSettings()
