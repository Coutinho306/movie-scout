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

- `--rebuild` — drop and recreate this variant's Qdrant collections before loading.
- `--refresh-taste` — recompute the taste profile even if it already exists.
- `--skip-taste` — reuse the existing taste profile without recomputing.

**Experiment variants** — each variant writes to its own Qdrant collection so
multiple can coexist for eval:

- `--embedder {openai-3-small,openai-3-large,minilm}` — choose the embedding
  model. Defaults to `openai-3-small` (or `EMBEDDER` env var).
- `--chunk-max-tokens N` — max tokens per review chunk (default 300 or
  `CHUNK_MAX_TOKENS` env var).
- `--chunk-overlap-tokens N` — token overlap between chunks (default 50 or
  `CHUNK_OVERLAP_TOKENS` env var).
- `--drop-variant` — delete this variant's two Qdrant collections and exit (no
  re-ingest).

Collection names are derived automatically:
- `openai-3-small` → `tmdb_movies__3small` / `tmdb_reviews__3small`
- `openai-3-large` → `tmdb_movies__3large` / `tmdb_reviews__3large`
- `minilm` → `tmdb_movies__minilm_c{max}o{overlap}` / `tmdb_reviews__minilm_c{max}o{overlap}`

Example: run a MiniLM variant with smaller chunks, then compare to the default:

```bash
uv run python3 -m ingestion.pipeline --embedder minilm --chunk-max-tokens 200 --skip-taste
```

Requires a populated `.env` (see `.env.example`).