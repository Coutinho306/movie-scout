# Embedding / chunk calibration report

Date: 2026-07-02
Runs: `eval/runs/retrieval_20260702T051634.csv` (round 1), `eval/runs/retrieval_20260702T163858.csv` (round 2)

## Question

Which embedding model + embed_text recipe + chunk params should the production ingest use?

## Evaluation method

**Golden-set retrieval eval** (offline, no LLM judge):

- **Queries**: 29 natural-language queries generated once from the personal watchlist
  (`eval/golden.py`, cached in `data/golden_set.json`). Each query has exactly one
  target `tmdb_id` as ground truth (avg 1.0 targets/query).
- **Corpus**: fixed calibration sample = golden targets + 300 distractors
  (`ingestion/scripts/build_calibration_sample.py`), 329 movies + 1,671 review chunks,
  ingested per variant into disposable `calib_`-prefixed Qdrant collections
  (`run_experiment --golden-sample`). Production collections untouched.
- **Isolation**: each variant is a separate collection; the query is embedded with the
  **same embedder** as the collection (pinned via `Settings.from_variant_suffix`,
  `eval/runs/retrieval_grid.py:47`) so query and documents share a vector space.
- **Retrieval path**: pure dense `search_movies` — hybrid, rerank, query_rewrite all off.
  Those knobs are calibrated separately post-ingest (they don't require re-embedding).
- **Metrics** (mean over 29 queries, computed in `eval/metrics/retrieval.py`):
  - `precision@k` — fraction of top-k that are relevant (ceiling 1/k with 1 target)
  - `recall@k` — fraction of targets found in top-k (with 1 target: hit rate)
  - `MRR` — 1/rank of first relevant hit
  - `nDCG@k` — rank-discounted gain; **decision metric**
  - `latency_p50_ms` — end-to-end search latency median
- **Grid**: variant × top_k [5, 10] (`eval/grids/calibration.yaml`), runner
  `eval/runs/retrieval_grid.py`, best config saved to `eval/runs/best_retrieval.json`.

## Variants tested

| Variant | Embedder | dim | Chunk (max/overlap) | embed_text | Round |
|---|---|---|---|---|---|
| `calib_3small_c300o50` | text-embedding-3-small | 1536 | 300/50 | base | 1 |
| `calib_3small_c300o50_keywords` | text-embedding-3-small | 1536 | 300/50 | base+keywords | 1, 2 |
| `calib_minilm_c300o50_keywords` | all-MiniLM-L6-v2 (local) | 384 | 300/50 | base+keywords | 1 |
| `calib_3large_c300o50_keywords` | text-embedding-3-large | 3072 | 300/50 | base+keywords | 2 |
| `calib_bgesmall_c300o50_keywords` | BAAI/bge-small-en-v1.5 (local, query-prefixed) | 384 | 300/50 | base+keywords | 2 |
| `calib_3small_c150o30_keywords` | text-embedding-3-small | 1536 | 150/30 | base+keywords | 2 |
| `calib_3small_c600o100_keywords` | text-embedding-3-small | 1536 | 600/100 | base+keywords | 2 |

## Results (top_k = 10)

| Variant | nDCG@10 | MRR | recall@10 | latency p50 (ms) |
|---|---|---|---|---|
| **3-large + keywords** | **0.626** | **0.593** | 0.724 | 1064 |
| 3-small + keywords (baseline) | 0.586 | 0.531 | **0.759** | 682 |
| 3-small + keywords, chunks 150/30 | 0.586 | 0.531 | 0.759 | 685 |
| 3-small + keywords, chunks 600/100 | 0.586 | 0.531 | 0.759 | 678 |
| bge-small + keywords | 0.373 | 0.308 | 0.586 | 2941 |
| minilm + keywords (round 1) | 0.286 | 0.225 | 0.483 | 2888 |
| 3-small base, no keywords (round 1) | 0.466 | 0.416 | 0.621 | 721 |

(top_k=5 ordering identical; full numbers in the CSVs.)

## Findings

1. **Keywords recipe remains the biggest lever** (+0.12 nDCG over base at same embedder;
   round 1 finding, confirmed).
2. **3-large beats 3-small** (+0.04 nDCG, +0.06 MRR) but pays ~6.5× embedding price and
   +55% query latency (larger vector, bigger payloads). Recall@10 slightly *lower*
   (0.724 vs 0.759) — 3-large ranks hits higher but misses a couple 3-small finds.
3. **Local embedders not competitive**: bge-small-en-v1.5 clearly better than MiniLM
   (0.37 vs 0.29 nDCG) but still ~0.21 nDCG below 3-small, at ~4× query latency
   (CPU encode). No free embedder tested reaches API quality here.
4. **Chunk size has zero effect on movie retrieval — by construction.** Chunking applies
   only to review chunks (`ingestion/chunking.py`); movie embed_text is one unchunked
   doc per film, and this eval measures `search_movies` only. The three chunk variants
   produced bit-identical metrics, empirically confirming the spike's "chunking is not
   the bottleneck" conclusion (`specs/spikes/ingest-chunk-embed-params/SPIKE.md`).
   Chunk params can only matter once a **review-retrieval** golden set exists.

## Decision

**Keep `text-embedding-3-small` + keywords recipe + chunks 300/50 as production default.**
3-large's +0.04 nDCG does not justify 6.5× embedding cost and +55% query latency for
this corpus size; revisit if ranking quality becomes the binding constraint at scale.
Keywords recipe is adopted (the actual win). Chunk 300/50 unchanged (no evidence any
size differs for movie retrieval).

## Reproduce

```bash
# ingest a variant into calib_ collections (sample only)
uv run python3 -m ingestion.scripts.run_experiment --embedder openai-3-large \
  --embed-text-recipe keywords --golden-sample --skip-taste

# run the grid
uv run python3 -m eval.runs.retrieval_grid --grid eval/grids/calibration.yaml
```
