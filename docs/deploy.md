# Deploy — Railway

Movie Scout deploys to [Railway](https://railway.app): the FastAPI backend and the
Streamlit frontend each run as a Railway service built from their Dockerfiles,
Postgres is Railway-managed, and Qdrant runs on [Qdrant Cloud](https://cloud.qdrant.io)
(Railway hosts Qdrant poorly). Qdrant Cloud is used for local runs too — `QDRANT_URL`
/ `QDRANT_API_KEY` in `.env` point at the cluster in both cases.

`railway.json` at the repo root configures the **api** service (Dockerfile build +
start command). The **frontend** service is added separately (below) pointing at
`Dockerfile.frontend`.

## Prerequisites
- Railway account + CLI: `npm i -g @railway/cli` then `railway login`.
- A Qdrant Cloud cluster with the `tmdb_movies` / `tmdb_reviews` collections
  ingested (run ingestion once against the cloud `QDRANT_URL`).

## Steps

```bash
# 1. New project, linked to this repo dir
railway init

# 2. Add Railway-managed Postgres (creates DATABASE_URL automatically)
railway add --plugin postgresql

# 3. Set secrets (never commit these)
railway variables set \
  OPENAI_API_KEY=... \
  TMDB_API_KEY=... \
  TAVILY_API_KEY=... \
  QDRANT_URL=https://<your-cluster>.qdrant.io:6333 \
  QDRANT_API_KEY=... \
  ALLOWED_ORIGINS=https://<frontend-domain> \
  LANGCHAIN_TRACING_V2=true \
  LANGCHAIN_API_KEY=...

# 4. Deploy the API (uses railway.json → Dockerfile.api)
railway up

# 5. Add the frontend as a second service
#    In the Railway dashboard: New Service → Deploy from repo →
#    set Dockerfile path to Dockerfile.frontend, and set:
#      API_BASE_URL = https://<api-service-domain>
#      TMDB_API_KEY = ...
```

## Env-var reference

| Variable | Service(s) | Notes |
|---|---|---|
| `DATABASE_URL` | api | auto-set by the Railway Postgres plugin |
| `OPENAI_API_KEY` | api | LLM + embeddings |
| `TMDB_API_KEY` | api, frontend | metadata + posters |
| `TAVILY_API_KEY` | api | web-search fallback |
| `QDRANT_URL` | api | Qdrant Cloud cluster URL |
| `QDRANT_API_KEY` | api | Qdrant Cloud key |
| `ALLOWED_ORIGINS` | api | frontend domain for CORS |
| `API_BASE_URL` | frontend | api service public URL |
| `LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY` | api | optional LangSmith traces |

## Notes
- The API image preloads the cross-encoder rerank model at build, so the first
  request isn't slow.
- Postgres schema is bootstrapped by `infra/postgres/init/01-schema.sql` locally;
  on Railway-managed Postgres, apply `infra/postgres/schema.sql` once via
  `railway connect postgres` (or `psql "$DATABASE_URL" -f infra/postgres/schema.sql`).
