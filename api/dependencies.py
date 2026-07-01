"""FastAPI dependency providers — overridable in tests via app.dependency_overrides."""

from __future__ import annotations

from functools import lru_cache
from typing import Callable

import asyncpg
from fastapi import Request

from agent.config import AgentSettings
from agent.main import run as agent_run
from agent.state import AgentRunResult
from api.config import ApiSettings


@lru_cache
def get_api_settings() -> ApiSettings:
    return ApiSettings()


@lru_cache
def get_agent_settings() -> AgentSettings:
    return AgentSettings()


def get_pg_pool(request: Request) -> asyncpg.Pool | None:
    """Return the pool created at startup, or None when persistence is disabled."""
    return getattr(request.app.state, "pg_pool", None)


def get_agent_run_fn() -> Callable[..., AgentRunResult]:
    """Return the agent entrypoint. Overridden in tests with a stub."""
    return agent_run
