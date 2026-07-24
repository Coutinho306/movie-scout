# Movie Scout

> Personal movie recommendation agent grounded in your Letterboxd taste — 
> RAG over TMDB metadata + reviews, with watchlist as held-out eval ground truth.

**Stack:** LangGraph · Qdrant · OpenAI · FastAPI · Streamlit  
**Capstone project for [LLM Zoomcamp](https://github.com/DataTalksClub/llm-zoomcamp) (2026 cohort)**

## Quickstart (Docker)

The runtime stack — Qdrant, API, UI, Postgres, Grafana — runs from one
compose file. Qdrant runs as a **local container** by default (no external
account needed); see [Deploy](#deploy) for pointing at Qdrant Cloud instead
when shipping somewhere other than your own machine.

```bash
# 1. configure
cp .env.example .env        # fill in TMDB_API_KEY, OPENAI_API_KEY, TAVILY_API_KEY
                            # QDRANT_URL=http://localhost:6333, QDRANT_API_KEY="" (local, keyless)
                            # (LANGCHAIN_API_KEY optional, for traces)

# 2. drop your Letterboxd export under data/letterboxd_export/

# 3. start Qdrant, then ingest once (~10 min: builds taste profile, loads Qdrant)
docker compose up -d qdrant
docker compose --profile ingest run --rm ingest      # or: make ingest

# 4. bring up the runtime stack
docker compose up -d                                  # or: make up

# 5. open the app
#    chat UI     → http://localhost:8501
#    Grafana     → http://localhost:3000  (anonymous viewer)

# 6. (optional) run the eval grids
make eval
```

Images (CPU-only torch — no CUDA): `Dockerfile.api` ~2.75 GB (backend),
`Dockerfile.frontend` ~780 MB (slim — Streamlit only, no agent/torch),
`Dockerfile.ingest` ~2.75 GB (one-shot embedding job). The API/ingest bulk
is CPU PyTorch + sentence-transformers for the local embedder (`LocalEmbedder`).

`QDRANT_URL` / `QDRANT_API_KEY` are required either way. Point them at the
local container (`http://localhost:6333`, empty key) for development, or at a
managed [Qdrant Cloud](https://cloud.qdrant.io) cluster for deploy — see
[`docs/deploy.md`](docs/deploy.md).

## Run the app (without Docker)

Two processes on the host. Both read `.env`.

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

## Testing & evaluation

Beyond the automated retrieval grids (`make eval`), the agent was manually
tested end-to-end across multiple rounds — real `/ask` queries, reading the
actual output, root-causing what broke. This caught several agent-logic bugs
(self-recommendation, missing exact-match search, silent title-collision
resolution) that a retrieval-only metric can't see. See
[`docs/manual_testing.md`](docs/manual_testing.md).

Retrieval quality on abstract queries is a known, documented open issue —
see [`docs/retrieval_quality.md`](docs/retrieval_quality.md) for the root
cause, what was tried, and why it's reported openly rather than hidden.

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
| `openai-3-large` | `tmdb_movies__3large_c300o50` | `tmdb_reviews__3large_c300o50` |
| `minilm` (default chunks) | `tmdb_movies__minilm_c300o50` | `tmdb_reviews__minilm_c300o50` |
| `minilm --chunk-max-tokens 200` | `tmdb_movies__minilm_c200o50` | `tmdb_reviews__minilm_c200o50` |

Additional flags:

- `--embedder {openai-3-small,openai-3-large,minilm,bge-small}` — embedding model.
- `--chunk-max-tokens N` — max tokens per review chunk (default 300).
- `--chunk-overlap-tokens N` — token overlap between chunks (default 50).
- `--rebuild` — drop and recreate this variant's collections before loading.
- `--drop-variant` — delete this variant's collections and exit.

## Retrieval layers

`retrieval/movies.py` supports independent knobs (`retrieval/config.py`) on
top of plain dense vector search:

| Knob | What it does | Layer it changes | Shipped default | Why |
|---|---|---|---|---|
| `query_rewrite` | HyDE: generates a hypothetical answer, embeds it, blends with the raw query vector before search (`retrieval/hyde.py`). | Query embedding, before retrieval. | **`True`** (on) | Wins the grid outright: +0.027–0.043 nDCG in every config it's paired with. The only knob that consistently helps as a static default. |
| `hybrid` | Fuses dense (vector) search with BM25 lexical search via Qdrant RRF, instead of dense-only. | Candidate retrieval (replaces the single dense query). | **Routed dynamically**, not a static default (see below) | A fixed on/off setting loses on the golden set's query mix, but the agent doesn't use a fixed setting — see next paragraph. |
| `top_k` | How many results are returned/considered. | Result-list size, both ends. | `10` | Grid tests 5 vs 10; ties on nDCG at the winning config, so left at the higher-recall value. |

**Hybrid search is smart-routed, not a static flag.** The static `hybrid`
default in `retrieval/config.py` is `False` and that's what the CLI/grid tools
use unless told otherwise, but the agent's real search tool
(`agent/tools/vector_search_movies.py:56`) doesn't read that static default —
it calls `classify_query_mode(query)` per query and turns hybrid on only for
queries that look like genre/cast/mood lookups, off for exact-title or
abstract queries. The 120-query tiered diagnostic
(`hybrid_search_eval.md`) shows why this routing exists: hybrid helps a lot on
genre/cast-word queries (tier 2, +0.396 nDCG) but ties or dilutes on exact
titles and abstract queries — so a query-aware router beats a fixed on/off
setting, and that's what ships.

`classify_query_mode` (`agent/tools/query_mode.py`) is a deterministic
regex/lexical classifier — no LLM call — mapping each query shape to a tier:

| Tier | Query shape | Example | Mode |
|---|---|---|---|
| 0 | Verbatim title, ≤4 tokens, no genre/sentence structure | `"Knives Out"` | dense |
| 1 | Narrative/overview sentence, >8 tokens | `"A detective investigates his wealthy patriarch's death..."` | hybrid |
| 2 | Templated `"a/an ... film — mood"` shape (genre + structure) | `"a heist film — stylish and tense"` | hybrid |
| 3 | Conversational/abstract request prefix | `"I'm looking for something uplifting"` | dense |

Dense-biased on uncertainty: an unmatched shape defaults to dense, since a
false "hybrid" on an abstract query costs more recall than a false "dense"
merely forgoes lift.

**Re-ranking (cross-encoder) was removed.** Two models were measured — the
original `ms-marco-MiniLM-L-6-v2` and a short-text candidate swap
`stsb-distilroberta-base` — and both were a net loss on nDCG across every
query bucket. The real bottleneck is a recall gap (~0.14 @k5): the right
films are frequently not in the candidate pool, so reordering cannot help.
Full numbers and verdict are in [`eval/runs/rerank_eval.md`](eval/runs/rerank_eval.md).
Re-tested after the BM25 keywords upgrade (below) in case the wider keyword-boosted
candidate pool changed the calculus — it didn't: still a net loss on every config.
See [`eval/runs/rerank_retest_post_keywords.md`](eval/runs/rerank_retest_post_keywords.md).

## Eval

Offline evaluation grid-searches retrieval and LLM knobs against a ground-truth
set built from the held-out watchlist (`data/letterboxd_export/watchlist.csv`).

Embedder / chunk / embed_text calibration (method, results, and the decision to
ship 3-small + keywords) is documented in
[`eval/runs/calibration_report.md`](eval/runs/calibration_report.md); the
decision is also logged in [`DECISIONS.md`](DECISIONS.md).

Hybrid (dense + BM25) search evaluation — sparse-index enrichment,
before/after nDCG@10 by query-difficulty tier — is documented in
[`eval/runs/hybrid_search_eval.md`](eval/runs/hybrid_search_eval.md).

Re-ranking evaluation — two cross-encoder models measured against the golden
set, both net loss; the decision to remove reranking is documented in
[`eval/runs/rerank_eval.md`](eval/runs/rerank_eval.md).

### Step 1 — run retrieval grid

```bash
uv run python3 -m eval.cli retrieval
```

Reads `eval/grids/retrieval.yaml` and runs every cartesian combination of
`top_k`, `variant`, `hybrid`, and `query_rewrite`. Writes results to
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
the agent). Open the `movie_scout` project to inspect a run's node graph, token
usage, and latency per step.

## Deploy

The API and UI deploy to [Railway](https://railway.app) (managed Postgres +
Qdrant Cloud). Full steps, env-var table, and the `railway.json` build config are
in [`docs/deploy.md`](docs/deploy.md).

<!-- Live demo URL + screenshot added after first deploy. -->
