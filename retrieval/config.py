"""Retrieval-time settings — knobs that don't affect ingestion."""

from __future__ import annotations

from pydantic import PrivateAttr
from pydantic_settings import BaseSettings, SettingsConfigDict

from ingestion.config import Settings as IngestionSettings


class RetrievalSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    top_k: int = 10
    hybrid: bool = False
    query_rewrite: bool = False
    score_threshold: float | None = None
    taste_alpha: float = 0.5  # weight on retrieval_score in blended rank

    # Optional override so callers (e.g. the calibration grid) can pin the exact
    # ingestion config a collection was built with. When set, the query embeds in
    # that variant's space instead of the .env default — without it, a minilm
    # collection would be queried with a 3-small vector (dimension mismatch).
    _ingestion_override: IngestionSettings | None = PrivateAttr(default=None)

    def with_ingestion(self, ingestion: IngestionSettings) -> "RetrievalSettings":
        """Return a copy pinned to a specific ingestion config."""
        clone = self.model_copy()
        clone._ingestion_override = ingestion
        return clone

    def ingestion(self) -> IngestionSettings:
        """Return the IngestionSettings to use — the override if pinned, else .env."""
        if self._ingestion_override is not None:
            return self._ingestion_override
        return IngestionSettings()
