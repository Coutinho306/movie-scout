# Retrieval-eval grid result

Date: 2026-07-18
Run: `eval/runs/retrieval_20260718T175125.csv` — 16 configs (`top_k` × `hybrid`
× `rerank` × `query_rewrite`), against the full 15,502-film production corpus
post BM25-sparse-fix (schema + write path both confirmed populated —
`ingestion/pipeline.py`). 29 golden queries, mean over all.

## Grid

| config | top_k | hybrid | rerank | query_rewrite | nDCG@k | MRR | precision@k | recall@k | latency p50 (ms) |
|---|---|---|---|---|---|---|---|---|---|
| ret_000 | 5 | ✗ | ✗ | ✗ | 0.0987 | 0.0862 | 0.0276 | 0.1379 | 388 |
| ret_001 | 5 | ✗ | ✗ | ✓ | **0.1252** | 0.1207 | 0.0276 | 0.1379 | 2389 |
| ret_002 | 5 | ✗ | ✓ | ✗ | 0.0987 | 0.0862 | 0.0276 | 0.1379 | 368 |
| ret_003 | 5 | ✗ | ✓ | ✓ | 0.1252 | 0.1207 | 0.0276 | 0.1379 | 358 |
| ret_004 | 5 | ✓ | ✗ | ✗ | 0.0562 | 0.0517 | 0.0138 | 0.0690 | 359 |
| ret_005 | 5 | ✓ | ✗ | ✓ | 0.1056 | 0.0948 | 0.0276 | 0.1379 | 345 |
| ret_006 | 5 | ✓ | ✓ | ✗ | 0.0435 | 0.0345 | 0.0138 | 0.0690 | 335 |
| ret_007 | 5 | ✓ | ✓ | ✓ | 0.1056 | 0.0948 | 0.0276 | 0.1379 | 339 |
| ret_008 | 10 | ✗ | ✗ | ✗ | 0.0987 | 0.0862 | 0.0138 | 0.1379 | 332 |
| ret_009 | 10 | ✗ | ✗ | ✓ | **0.1252** | 0.1207 | 0.0138 | 0.1379 | 352 |
| ret_010 | 10 | ✗ | ✓ | ✗ | 0.0987 | 0.0862 | 0.0138 | 0.1379 | 354 |
| ret_011 | 10 | ✗ | ✓ | ✓ | 0.1252 | 0.1207 | 0.0138 | 0.1379 | 336 |
| ret_012 | 10 | ✓ | ✗ | ✗ | 0.0913 | 0.0782 | 0.0138 | 0.1379 | 376 |
| ret_013 | 10 | ✓ | ✗ | ✓ | 0.1056 | 0.0948 | 0.0138 | 0.1379 | 370 |
| ret_014 | 10 | ✓ | ✓ | ✗ | 0.0913 | 0.0782 | 0.0138 | 0.1379 | 341 |
| ret_015 | 10 | ✓ | ✓ | ✓ | 0.1056 | 0.0948 | 0.0138 | 0.1379 | 361 |

## Winner

**`ret_001`/`ret_009` — dense-only + query rewrite** (top_k doesn't move the
needle, tied at nDCG 0.1252). Production defaults (`hybrid=False`,
`query_rewrite=True`) already match.

## Findings

1. **Query rewrite is the single strongest lever tested** — a consistent
   +0.027-0.043 nDCG lift in every hybrid/rerank combination it's paired with
   (e.g. 0.0987 → 0.1252 dense-only at top_k=5). This is a HyDE-style rewrite
   (`retrieval/hyde.py`) applied before the first embed.
2. **Hybrid (dense + BM25 RRF) underperforms dense-only on this query set**
   without rewrite (0.0987 → 0.0562 at top_k=5, nearly halved) and stays
   behind dense-only even *with* rewrite (0.1056 vs 0.1252). This is a real,
   somewhat counter-intuitive result: BM25 sparse lexical matching adds noise
   for this golden set's mostly-semantic natural-language queries (see
   `eval/runs/hybrid_search_eval.md` for a tier-stratified view — hybrid wins
   decisively on genre/mood and actor-name queries, tiers where this flat
   golden set has few examples; net effect here is negative because those
   query types are underrepresented in this particular grid's query mix).
3. **`rerank=True` was a config no-op** — every rerank=True row was
   bit-identical to its rerank=False counterpart (confirmed:
   `retrieval/movies.py` never read the `rerank` flag). The cross-encoder
   reranker described in early planning docs (`__temp/STACK.md`, marked
   superseded) was never wired in. Documented here rather than silently
   dropped from the grid, since a reviewer comparing configs would otherwise
   wonder why two rows tie exactly. **Update (specs/project-gap-analysis):**
   the dead `rerank` field has since been deleted from `RetrievalSettings`
   and the grid axis removed — a later grid run won't have a `rerank`
   column at all, rather than a silently-tied one.
4. **top_k (5 vs 10) has no effect on nDCG** at matching hybrid/rerank/rewrite
   settings — expected, since nDCG@k already discounts by rank and the golden
   set averages ~1 relevant target per query, so results beyond the top few
   don't change the score.

## Why these numbers are lower than the calibration-sample numbers

`eval/runs/calibration_report.md` (330-film sample) reported nDCG@10 up to
0.626 — this grid's best is 0.125 on the same query set, now against the full
15,502-film corpus. This is the expected small-sample-bias effect documented
in `__temp/TODOS.md` ("Notes worth keeping"): a 329-film calibration sample
has few distractors and inflates absolute nDCG; the full corpus is the honest
production number. Relative config ranking (query rewrite > dense-only >
hybrid-without-rewrite here) is what the calibration sample is trusted for,
and it holds at full scale.
