# retrieval/

Low-level search primitives over the Qdrant collections populated by `ingestion/`.
Called by the agent (spec 0005) and the eval harness (spec 0006).

## Quickstart

```python
from retrieval.config import RetrievalSettings
from retrieval.movies import search_movies
from retrieval.reviews import search_reviews
from retrieval.taste import score_against_taste
from retrieval.rerank import cross_encode_rerank
from retrieval.rewrite import rewrite_query

settings = RetrievalSettings(top_k=10, hybrid=False, rerank=False, query_rewrite=False)

query = "slow meditative film about memory"

# Optional: rewrite query for better recall
query = rewrite_query(query)

# Search
movies = search_movies(query, settings=settings)
reviews = search_reviews(query, settings=settings)

# Blend with taste profile
movies = score_against_taste(movies)   # loads data/taste_profile.json automatically

# Rerank by cross-encoder
reviews = cross_encode_rerank(query, reviews)
```

## Flag matrix

| Flag | Default | Effect |
|---|---|---|
| `top_k` | 10 | Number of results from Qdrant |
| `hybrid` | False | BM25 + dense RRF (degrades to vector-only if no sparse index) |
| `rerank` | False | Cross-encoder rerank (use in caller, not auto-applied) |
| `query_rewrite` | False | LLM rewrites query before search (use in caller) |
| `score_threshold` | None | Minimum similarity score filter |
| `taste_alpha` | 0.5 | Weight of retrieval score vs taste score in blended rank |

## Environment variables

| Var | Required | Default |
|---|---|---|
| `QDRANT_URL` | yes | — |
| `QDRANT_API_KEY` | yes | — |
| `OPENAI_API_KEY` | yes | — |
| `MODEL_ORCHESTRATOR` | no | `gpt-4o-mini` (for query rewrite) |
| `EMBEDDER` | no | `openai-3-small` |

## Notes

- `hybrid=True` attempts Qdrant native RRF fusion. Falls back to dense-only with a
  warning when the collection lacks a sparse vector field.
- `exclude_tmdb_ids` in `MovieFilters` is applied in Python post-fetch (Qdrant
  KEYWORD index on integer tmdb_id doesn't support MatchExcept reliably).
- `rewrite_query` caches results per (query, model) for the process lifetime.
- Cross-encoder model loads once per process via `functools.lru_cache`.
