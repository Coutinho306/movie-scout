"""Postgres persistence for /ask runs and /feedback rows.

Table DDL lives in spec 0009's bootstrap script — this module only INSERTs.
Every write is best-effort: a failure is logged and swallowed so it never
fails the user-facing request.
"""

from __future__ import annotations

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
    session_id: str | None,
    result: AgentRunResult,
) -> None:
    """Persist one /ask run. Best-effort — logs and swallows any failure."""
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO runs (
                    run_id, session_id, query, final_answer,
                    latency_ms, cost_usd, tool_calls,
                    orchestrator_turns, rag_calls, web_calls
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                """,
                run_id,
                session_id,
                query,
                result.final_answer,
                result.latency_ms,
                result.cost_usd,
                result.tool_calls,
                result.orchestrator_turns,
                result.rag_calls,
                result.web_calls,
            )
    except Exception:  # noqa: BLE001 — persistence must never break the request
        logger.exception("insert_run failed for run_id=%s", run_id)


async def insert_feedback(
    pool: asyncpg.Pool,
    run_id: UUID,
    rating: str,
    comment: str | None,
) -> None:
    """Persist one /feedback row. Best-effort — logs and swallows any failure."""
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO feedback (run_id, rating, comment)
                VALUES ($1, $2, $3)
                """,
                run_id,
                rating,
                comment,
            )
    except Exception:  # noqa: BLE001 — persistence must never break the request
        logger.exception("insert_feedback failed for run_id=%s", run_id)
