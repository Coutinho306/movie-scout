"""Postgres persistence for /ask runs and /feedback rows.

Table DDL lives in spec 0009 (infra/postgres/schema.sql: agent_runs /
agent_feedback) — this module only INSERTs. Every write is best-effort: a
failure is logged and swallowed so it never fails the user-facing request.
"""

from __future__ import annotations

import json
import logging
from uuid import UUID

import asyncpg

from agent.state import AgentRunResult

logger = logging.getLogger(__name__)


async def init_pool(database_url: str) -> asyncpg.Pool:
    """Create an asyncpg connection pool for the given DATABASE_URL."""
    return await asyncpg.create_pool(dsn=database_url)


async def insert_run(
    pool: asyncpg.Pool,
    run_id: UUID,
    query: str,
    result: AgentRunResult,
    *,
    model: str | None = None,
    prompt_variant: str | None = None,
) -> None:
    """Persist one /ask run into agent_runs. Best-effort — logs, never raises."""
    citations = json.dumps([c.model_dump() for c in result.citations])
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO agent_runs (
                    run_id, user_query, final_answer,
                    latency_ms, cost_usd, tool_calls,
                    rag_calls, web_calls, orchestrator_turns,
                    model, prompt_variant, citations
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12::jsonb
                )
                """,
                run_id,
                query,
                result.final_answer,
                result.latency_ms,
                result.cost_usd,
                result.tool_calls,
                result.rag_calls,
                result.web_calls,
                result.orchestrator_turns,
                model,
                prompt_variant,
                citations,
            )
    except Exception:  # noqa: BLE001 — persistence must never break the request
        logger.exception("insert_run failed for run_id=%s", run_id)


async def insert_feedback(
    pool: asyncpg.Pool,
    run_id: UUID,
    rating: str,
    comment: str | None,
) -> None:
    """Persist one /feedback row into agent_feedback. Best-effort — never raises."""
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO agent_feedback (run_id, rating, comment)
                VALUES ($1, $2, $3)
                """,
                run_id,
                rating,
                comment,
            )
    except Exception:  # noqa: BLE001 — persistence must never break the request
        logger.exception("insert_feedback failed for run_id=%s", run_id)
