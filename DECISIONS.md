# Decisions

Log of architecture tradeoffs and choices made during development.

| Date | Decision | Rationale | Alternatives considered |
|------|----------|-----------|------------------------|
| 2026-07-02 | Keep text-embedding-3-small + keywords recipe + chunks 300/50 after calibration round 2 (see eval/runs/calibration_report.md) | 3-large gains only +0.04 nDCG for 6.5× embed cost and +55% query latency; keywords recipe is the real lever (+0.12 nDCG); chunk size provably irrelevant to movie retrieval (movies unchunked) | 3-large (marginal gain, kept as runner-up collection), bge-small-en-v1.5 / MiniLM (local, nDCG 0.37/0.29 vs 0.59 — not competitive), chunk 150/30 and 600/100 (bit-identical metrics) |
| 2026-06-12 | Use Letterboxd CSV export as source-of-truth | API denied for AI/recommendation use cases; manual export is the only legitimate path | Letterboxd API (denied), HTML scraping (TOS prohibits) |
| 2026-06-12 | TMDB API as primary metadata source | Official API, free, structured JSON, covers metadata + reviews + providers | OMDb (no reviews), IMDb (TOS prohibits scraping) |
| 2026-06-12 | Tavily as web search fallback only | RAG indexed first; web search adds 2-5s latency and eval unpredictability | Primary web search (rejected — latency, cost, eval issues) |
| 2026-06-12 | Reject scraping IMDb, Rotten Tomatoes, Metacritic, Letterboxd reviews | All TOS explicitly prohibit scraping for AI use | Scraping with custom user-agent (rejected — ethical and reputational risk) |
| 2026-06-12 | LangChain + LangGraph as agent framework | Familiar from Brewmaster, no learning curve, ships faster under deadline | Pydantic AI (deferred — type safety attractive but adds learning time) |
| 2026-06-12 | Qdrant as vector DB | Used in Zoomcamp curriculum, supports hybrid search, easy Docker setup | pgvector (less flexibility for hybrid search) |
| 2026-06-12 | Multiple Qdrant collections separated by source | Heterogeneous chunking + retrieval strategies per source; easier eval per source | Single collection with metadata filtering (chunking compromised) |
| 2026-06-12 | User taste profile NOT in Qdrant | 108 films won't be vector-searched against each other; taste matching is deterministic centroid math | Separate Qdrant collection for user films (unnecessary complexity) |
| 2026-06-29 | Plain Python for ingestion (removed dlt) | dlt was bypassed by direct qdrant-client upserts + write_disposition=replace; none of its value (incremental/schema/dedup) applied — it only wrote status rows to a throwaway duckdb. Same 2 rubric pts as a script, minus the explain-burden for a non-course tool. | dlt (removed — pure ceremony here), Airflow/Prefect (overkill) |
| 2026-06-12 | OpenAI GPT-4o-mini as LLM | Low cost (~$0.15/1M tokens) for iteration, good quality | GPT-4o (higher cost, marginal quality gain for use case) |
| 2026-06-12 | text-embedding-3-small for embeddings | Same SDK as LLM, no model infra to manage, cost negligible (~$0.05 total) | MiniLM (open source, free, but lower MTEB and infra to manage) |
| 2026-06-12 | FastAPI + Streamlit as interface | Zoomcamp standard, ships fast, reusable in future projects | Pure Streamlit (less API discipline), pure FastAPI (no demo UI) |
| 2026-06-12 | Grafana + Postgres for monitoring | Standard pattern, satisfies criterion "dashboard with 5+ charts" | Streamlit page (simpler but less professional), Prometheus (overkill) |
| 2026-06-12 | LangSmith for observability | Natural fit with LangChain/LangGraph | Logfire (better with Pydantic AI, not our choice) |
| 2026-06-12 | RAGAS + custom precision@k for eval | RAGAS standard for RAG; watchlist as personal ground truth = stronger than synthetic golden set | Synthetic golden set only (Brewmaster's weakness, avoided here) |
| 2026-06-12 | Watchlist as held-out test set | Personal ground truth, scalable, no synthetic data needed | Manual golden set (less defensible) |
| 2026-06-12 | Docker Compose for local orchestration | Standard for multi-service projects (Qdrant + Postgres + Grafana + app) | Kubernetes (overkill for project scope) |
| 2026-06-12 | Single agent with tools, not multi-agent | Linear routing fits use case; Brewmaster already covers multi-agent pattern | Multi-agent like Brewmaster (unnecessary complexity for this use case) |
| 2026-06-12 | Chunking varies by source (no fixed size) | Granularity follows source nature: metadata small, reviews medium, plots large | Fixed chunk size (compromised retrieval quality) |
| 2026-06-12 | BR-default region for streaming providers | LATAM moat, most movie recommenders default to US-only | US-default (less differentiated for target market) |