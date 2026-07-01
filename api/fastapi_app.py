"""FastAPI app — REST interface to the Movie Scout agent.

Routes:
    POST /ask       — run the agent for one query, return a narrowed result.
    POST /feedback  — persist thumbs up/down for a prior run.
    GET  /healthz   — liveness + cheap Qdrant/OpenAI connectivity flags.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Callable
from uuid import uuid4

from fastapi import Depends, FastAPI, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

from agent.config import AgentSettings
from agent.state import AgentRunResult
from api import store
from api.config import ApiSettings
from api.dependencies import (
    get_agent_run_fn,
    get_agent_settings,
    get_api_settings,
    get_pg_pool,
)
from api.models import AskRequest, AskResponse, FeedbackRequest

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Open the asyncpg pool at startup (if DATABASE_URL set); close it at shutdown."""
    settings = get_api_settings()
    app.state.pg_pool = None
    if settings.database_url:
        try:
            app.state.pg_pool = await store.init_pool(settings.database_url)
        except Exception:  # noqa: BLE001 — DB down must not block the API booting
            logger.exception("failed to init pg pool; persistence disabled")
    yield
    if app.state.pg_pool is not None:
        await app.state.pg_pool.close()


def create_app(settings: ApiSettings | None = None) -> FastAPI:
    settings = settings or get_api_settings()
    app = FastAPI(title="Movie Scout API", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.origins_list(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post("/ask", response_model=AskResponse)
    async def ask(
        req: AskRequest,
        agent_run: Callable[..., AgentRunResult] = Depends(get_agent_run_fn),
        agent_settings: AgentSettings = Depends(get_agent_settings),
        pool=Depends(get_pg_pool),
    ) -> AskResponse:
        run_id = uuid4()
        result: AgentRunResult = await run_in_threadpool(
            agent_run, req.query, agent_settings
        )

        logger.info(
            json.dumps(
                {
                    "step": "api_ask",
                    "run_id": str(run_id),
                    "latency_ms": round(result.latency_ms, 1),
                    "cost_usd": round(result.cost_usd, 6),
                    "tool_calls": result.tool_calls,
                }
            )
        )

        if pool is not None:
            asyncio.create_task(
                store.insert_run(
                    pool,
                    run_id,
                    req.query,
                    result,
                    model=agent_settings.model_agent,
                    prompt_variant=(
                        "rewrite" if agent_settings.query_rewrite else "baseline"
                    ),
                )
            )

        return AskResponse(
            run_id=run_id,
            final_answer=result.final_answer,
            citations=result.citations,
            latency_ms=result.latency_ms,
            cost_usd=result.cost_usd,
            tool_calls=result.tool_calls,
        )

    @app.post("/feedback", status_code=204)
    async def feedback(req: FeedbackRequest, pool=Depends(get_pg_pool)) -> Response:
        if pool is not None:
            await store.insert_feedback(pool, req.run_id, req.rating, req.comment)
        return Response(status_code=204)

    @app.get("/healthz")
    async def healthz() -> dict:
        return {
            "status": "ok",
            "qdrant": _qdrant_ok(),
            "openai": bool(os.environ.get("OPENAI_API_KEY")),
        }

    return app


def _qdrant_ok() -> bool:
    """Cheap Qdrant reachability check. Never raises — returns False on any error."""
    try:
        from retrieval.client import get_qdrant_client

        get_qdrant_client().get_collections()
        return True
    except Exception:  # noqa: BLE001 — probe flag only, never propagate
        return False


app = create_app()
