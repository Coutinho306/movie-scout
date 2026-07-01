# Movie Scout

> Personal movie recommendation agent grounded in your Letterboxd taste — 
> RAG over TMDB metadata + reviews, with watchlist as held-out eval ground truth.

**Stack:** LangGraph · Qdrant · OpenAI · FastAPI · Streamlit  
**Capstone project for [LLM Zoomcamp](https://github.com/DataTalksClub/llm-zoomcamp) (2026 cohort)**

## Run the app

Two processes: the FastAPI backend and the Streamlit UI. Both read `.env`.

```bash
# 1. backend (agent over HTTP)
uv run uvicorn api.fastapi_app:app --reload

# 2. UI (in a second terminal)
uv run streamlit run frontend/streamlit_app.py
```

The UI opens on http://localhost:8501 and calls the backend at `API_BASE_URL`
(default `http://localhost:8000`). Ask for something to watch, expand the
citation cards, and rate the answer with 👍 / 👎.

![Chat UI](docs/screenshots/chat.png)
![Feedback](docs/screenshots/feedback.png)

See [`frontend/README.md`](frontend/README.md) for env vars and details.

## Ingestion

Run the TMDB ingestion pipeline as a module from the project root (the package
uses absolute imports, so a path-based `python3 ingestion/pipeline.py` will not
resolve):

```bash
uv run python3 -m ingestion.pipeline
```

This is a single entry point: it first builds the taste profile
(`data/taste_profile.json`) from your Letterboxd export if it's missing, then
discovers and ingests candidate movies. Discovery genres are derived from your
taste (the genres of the films you rated/liked, weighted by rating).

Flags:

- `--rebuild` — drop and recreate the Qdrant collections before loading (clean slate).
- `--refresh-taste` — recompute the taste profile even if it already exists.
- `--skip-taste` — reuse the existing taste profile without recomputing.

Writes to the canonical collections `tmdb_movies` and `tmdb_reviews`.

Requires a populated `.env` (see `.env.example`).

## Experiments

To test a different embedder or chunking params without touching the canonical
collections, use the experiment runner:

```bash
uv run python3 -m ingestion.scripts.run_experiment --embedder minilm --chunk-max-tokens 200 --skip-taste
```

Each variant writes to its own auto-named Qdrant collection so variants coexist:

| Variant | Movies collection | Reviews collection |
|---|---|---|
| `openai-3-large` | `tmdb_movies__3large` | `tmdb_reviews__3large` |
| `minilm` (default chunks) | `tmdb_movies__minilm_c300o50` | `tmdb_reviews__minilm_c300o50` |
| `minilm --chunk-max-tokens 200` | `tmdb_movies__minilm_c200o50` | `tmdb_reviews__minilm_c200o50` |

Additional flags:

- `--embedder {openai-3-small,openai-3-large,minilm}` — embedding model.
- `--chunk-max-tokens N` — max tokens per review chunk (default 300).
- `--chunk-overlap-tokens N` — token overlap between chunks (default 50).
- `--rebuild` — drop and recreate this variant's collections before loading.
- `--drop-variant` — delete this variant's collections and exit.

## Eval

Offline evaluation grid-searches retrieval and LLM knobs against a ground-truth
set built from the held-out watchlist (`data/letterboxd_export/watchlist.csv`).

### Step 1 — run retrieval grid

```bash
uv run python3 -m eval.cli retrieval
```

Reads `eval/grids/retrieval.yaml` and runs every cartesian combination of
`top_k`, `variant`, `hybrid`, `rerank`, and `query_rewrite`. Writes results to
`eval/runs/retrieval_<ts>.csv` and prints the winning config by `mean_ndcg_at_k`.
The winner is saved to `eval/runs/best_retrieval.json` for the LLM grid.

### Step 2 — run LLM grid

```bash
uv run python3 -m eval.cli llm
```

Requires `eval/runs/best_retrieval.json` (errors clearly if missing). Pins
retrieval to the step-1 winner and varies `temperature` + prompt variant.
Judges answers with RAGAS `Faithfulness`, `AnswerRelevancy`, and a custom
`TasteMatch` score. Writes `eval/runs/llm_<ts>.csv`.

### Run both in order

```bash
uv run python3 -m eval.cli all
```

### Outputs

| File | Contents |
|---|---|
| `eval/runs/retrieval_<ts>.csv` | One row per retrieval config: precision, recall, MRR, nDCG, latency |
| `eval/runs/best_retrieval.json` | Winning retrieval config (written by retrieval grid) |
| `eval/runs/llm_<ts>.csv` | One row per LLM config: faithfulness, relevancy, taste_match |
| `data/golden_set.json` | Cached ground-truth queries (gitignored; regenerated if absent) |

### Reading the winning config

`best_retrieval.json` contains the config id and full parameter set of the
config with the highest `mean_ndcg_at_k`. Pass those parameters to the agent
or set them as defaults in `retrieval/config.py` before shipping.

## Monitoring

Every `/ask` run and every 👍/👎 is written to Postgres (`agent_runs`,
`agent_feedback`) and visualised in a provisioned Grafana dashboard.

```bash
docker compose up -d postgres grafana
```

Open http://localhost:3000 — the **Movie Scout — Monitoring** dashboard loads
automatically (anonymous viewer access is on; admin login is `admin` /
`GF_SECURITY_ADMIN_PASSWORD`, default `admin`). Schema is created on the
Postgres container's first boot from `infra/postgres/init/`.

To populate the dashboard without running the agent dozens of times, apply the
opt-in demo seed:

```bash
docker compose exec -T postgres \
  psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" < infra/postgres/seed_demo.sql
```

Panels:

1. **Total runs (7d)** — volume single-stat.
2. **Thumbs-up rate (7d)** — `up / (up+down)`.
3. **Runs per hour (24h)** — throughput time series.
4. **Latency p50 / p95 (24h)** — response-time percentiles.
5. **Cost USD per day (14d)** — spend trend.
6. **Top 10 recommended TMDB ids** — from `citations` JSONB.
7. **Last 20 thumbs-down** — query + comment for triage.

Dashboard JSON and datasource live in `infra/grafana/` and are provisioned on
container start — no manual import.

![Grafana dashboard](docs/screenshots/grafana.png)

### Traces

LLM traces go to [LangSmith](https://smith.langchain.com) — set
`LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY` in `.env` (already wired by
the agent, spec 0005). Open the `movie_scout` project to inspect a run's node
graph, token usage, and latency per step.