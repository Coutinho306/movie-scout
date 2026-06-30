"""Agent-time settings — model IDs, routing caps, region. Loaded from .env."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class AgentSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    model_agent: str = "gpt-4o-mini"
    model_orchestrator: str = "gpt-4o-mini"
    max_orchestrator_turns: int = 4
    tavily_max_results: int = 5
    watch_region: str = "BR"

    # Reasoning models reject response_format=json_object; disable JSON mode for them.
    reasoning_models: set[str] = {"o1", "o1-mini", "o3-mini"}
