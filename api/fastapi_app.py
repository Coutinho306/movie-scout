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
import time
from contextlib import asynccontextmanager
from typing import Callable
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Request, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

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
from api.models import AskRequest, AskResponse, FeedbackRequest, TasteProfileResponse

logger = logging.getLogger(__name__)

MAX_TASTE_UPLOAD_BYTES = 20 * 1024 * 1024  # 20MB — Letterboxd exports are typically <5MB


def _client_key(request: Request) -> str:
    """Rate-limit key. Behind a proxy (Railway) the real client IP is the first
    hop of X-Forwarded-For; fall back to the socket peer for direct/local runs."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return get_remote_address(request)


limiter = Limiter(key_func=_client_key)


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
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.origins_list(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post("/ask", response_model=AskResponse)
    @limiter.limit(settings.rate_limit)
    async def ask(
        request: Request,
        req: AskRequest,
        agent_run: Callable[..., AgentRunResult] = Depends(get_agent_run_fn),
        agent_settings: AgentSettings = Depends(get_agent_settings),
        pool=Depends(get_pg_pool),
    ) -> AskResponse:
        run_id = uuid4()

        # Build a per-request copy of settings carrying the caller's taste profile
        # and any franchise clarification data from the stateless round-trip (AC-4).
        # When no profile is present → cold start (profile=None → retrieval-only).
        per_request_settings = agent_settings.model_copy(
            update={
                "taste_profile": req.taste_profile,
                "clarification_answer": req.clarification_answer,
                "franchise_sibling_ids": list(req.franchise_sibling_ids or []),
            }
        )

        _t0 = time.perf_counter()
        try:
            result: AgentRunResult = await run_in_threadpool(
                agent_run, req.query, per_request_settings
            )
        except Exception:  # noqa: BLE001
            _elapsed_ms = (time.perf_counter() - _t0) * 1000
            logger.exception(
                json.dumps({"step": "api_ask_error", "run_id": str(run_id)})
            )
            return AskResponse(
                run_id=run_id,
                final_answer="Sorry, I couldn't complete that request — try rephrasing",
                citations=[],
                latency_ms=round(_elapsed_ms, 1),
                cost_usd=0.0,
                tool_calls=0,
                needs_clarification=False,
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

        # Skip analytics persistence on a clarify-pause (no recs generated yet;
        # the second /ask with the resolved answer will be the meaningful run).
        if pool is not None and not result.needs_clarification:
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
            needs_clarification=result.needs_clarification,
            clarification_question=result.clarification_question,
            franchise_sibling_ids=result.franchise_sibling_ids,
        )

    @app.post("/taste-profile", response_model=TasteProfileResponse)
    @limiter.limit("10/minute")
    async def taste_profile(
        request: Request,
        file: UploadFile,
    ) -> TasteProfileResponse:
        """Accept a Letterboxd ratings.csv or ZIP export; return an ephemeral TasteProfile.

        Writes nothing to disk or Postgres. Zero OpenAI embedding cost — vectors
        are pulled from the tmdb_movies corpus by point id.
        """
        from retrieval.taste_upload import build_taste_profile_from_upload

        tmdb_api_key = os.environ.get("TMDB_API_KEY", "")
        if not tmdb_api_key:
            raise HTTPException(
                status_code=503, detail="TMDB_API_KEY not configured on server"
            )

        filename = file.filename or ""
        data = await file.read()
        if len(data) > MAX_TASTE_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Upload exceeds {MAX_TASTE_UPLOAD_BYTES // (1024 * 1024)}MB limit",
            )

        try:
            result = await run_in_threadpool(
                build_taste_profile_from_upload,
                data,
                filename=filename,
                tmdb_api_key=tmdb_api_key,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            logger.exception("taste_profile upload failed")
            raise HTTPException(
                status_code=500, detail="Taste profile computation failed"
            ) from exc

        return TasteProfileResponse(
            profile=result.profile,
            resolved=result.report.resolved,
            tmdb_miss=result.report.tmdb_miss,
            out_of_corpus=result.report.out_of_corpus,
            total_input=result.report.total_input,
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
