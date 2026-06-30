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