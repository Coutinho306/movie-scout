"""Schema integration test — apply DDL to an ephemeral Postgres, exercise the join.

Marked `integration`: needs Docker for testcontainers. Skips with a reason when
Docker/testcontainers is unavailable so the default suite stays green.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

SCHEMA = Path(__file__).resolve().parents[2] / "infra" / "postgres" / "schema.sql"


@pytest.fixture
def pg_dsn():
    psycopg = pytest.importorskip("psycopg2")  # testcontainers' pg driver
    try:
        from testcontainers.postgres import PostgresContainer
    except ImportError:  # pragma: no cover
        pytest.skip("testcontainers not installed")

    try:
        with PostgresContainer("postgres:15") as pg:
            yield pg.get_connection_url(), psycopg
    except Exception as exc:  # noqa: BLE001 — Docker daemon likely unavailable
        pytest.skip(f"could not start Postgres container: {exc}")


def test_schema_and_join(pg_dsn) -> None:
    url, psycopg = pg_dsn
    # testcontainers hands back a SQLAlchemy-style URL; psycopg2 wants libpq form
    dsn = url.replace("postgresql+psycopg2://", "postgresql://")

    conn = psycopg.connect(dsn)
    conn.autocommit = True
    cur = conn.cursor()

    cur.execute(SCHEMA.read_text())

    run_id = str(uuid.uuid4())
    citations = json.dumps([{"tmdb_id": 655, "title": "Paris, Texas", "year": 1984}])
    cur.execute(
        """
        INSERT INTO agent_runs (run_id, user_query, final_answer, latency_ms,
            cost_usd, tool_calls, rag_calls, web_calls, orchestrator_turns,
            model, prompt_variant, citations)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
        """,
        (run_id, "slow film", "Watch Paris, Texas.", 123.4, 0.0012, 2, 2, 0, 1,
         "gpt-4o-mini", "baseline", citations),
    )
    cur.execute(
        "INSERT INTO agent_feedback (run_id, rating, comment) VALUES (%s, %s, %s)",
        (run_id, "up", "great"),
    )

    cur.execute(
        """
        SELECT r.user_query, f.rating, (r.citations->0->>'tmdb_id')::int
        FROM agent_feedback f JOIN agent_runs r ON r.run_id = f.run_id
        WHERE f.run_id = %s
        """,
        (run_id,),
    )
    row = cur.fetchone()
    assert row == ("slow film", "up", 655)

    conn.close()
