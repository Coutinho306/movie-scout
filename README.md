# Movie Scout

> Personal movie recommendation agent grounded in your Letterboxd taste — 
> RAG over TMDB metadata + reviews, with watchlist as held-out eval ground truth.

**Stack:** LangGraph · Qdrant · OpenAI · FastAPI · Streamlit  
**Capstone project for [LLM Zoomcamp](https://github.com/DataTalksClub/llm-zoomcamp) (2026 cohort)**

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