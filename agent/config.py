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
    prompt_variant: str = "v1"  # which agent/prompts/synthesize{_variant}.md to use

    # Per-request taste profile. When None, cold start (retrieval-only ordering).
    taste_profile: TasteProfile | None = Field(default=None, exclude=True)

    # Per-request franchise clarification answer echoed back from the client.
    # When present, the pre-graph gate skips detection and uses this to resolve
    # franchise inclusion/exclusion. Matches the stateless round-trip pattern of
    # taste_profile (AC-3, AC-7).
    clarification_answer: str | None = Field(default=None, exclude=True)

    # Franchise sibling tmdb_ids echoed back from the client on the second call.
    # Set from AskResponse.franchise_sibling_ids which the first /ask populates
    # when needs_clarification=True. On the second /ask, the client re-sends
    # these so the exclude path knows which IDs to filter (AC-6).
    franchise_sibling_ids: list[int] = Field(default_factory=list, exclude=True)

    # Reasoning models reject response_format=json_object; disable JSON mode for them.
    reasoning_models: set[str] = {"o1", "o1-mini", "o3-mini"}
