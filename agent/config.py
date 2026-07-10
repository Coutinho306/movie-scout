"""Agent-time settings — model IDs, routing caps, region. Loaded from .env."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from ingestion.models import TasteProfile


class AgentSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    model_agent: str = "gpt-4o-mini"
    model_orchestrator: str = "gpt-4o-mini"
    temperature: float = 0.0
    top_k: int = 10
    max_orchestrator_turns: int = 4
    tavily_max_results: int = 5
    watch_region: str = "BR"
    query_rewrite: bool = True  # rewrite the user query once before the first RAG retrieval

    # Per-request taste profile. When None, cold start (retrieval-only ordering).
    taste_profile: TasteProfile | None = Field(default=None, exclude=True)

    # Reasoning models reject response_format=json_object; disable JSON mode for them.
    reasoning_models: set[str] = {"o1", "o1-mini", "o3-mini"}
