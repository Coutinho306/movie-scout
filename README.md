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

- `--rebuild` — drop and recreate the Qdrant collections before loading (clean
  rebuild). Without it, reruns refresh in place (idempotent, no duplicates).
- `--refresh-taste` — recompute the taste profile even if it already exists.
- `--skip-taste` — reuse the existing taste profile without recomputing.

Requires a populated `.env` (see `.env.example`).