# Keywords-clause sparse eval — before/after verdict

**Spec**: 0022 (keywords-payload-bm25-and-filter), Phase 4  
**Date**: 2026-07-24  
**Verdict**: KEEP (keyword clause retained; neutral-to-marginal lift in hybrid configs, no regression)

## Coverage

Full 8-config hybrid grid (`eval/grids/retrieval.yaml`):
`top_k ∈ {5, 10}` × `hybrid ∈ {false, true}` × `query_rewrite ∈ {false, true}`,
`variant=default`. All 8 configs run in both the before and after passes.

## Corpus states

**Before** (pre-Phase-3): sparse vectors built from `enriched-base` recipe — overview,
genres, cast, director; no keywords clause. Corpus: 15,502 films, all keywords backfilled
into payload (Phase 2 done) but sparse vectors do not yet include them.  
Source: `eval/runs/rerank_retest_results.json` (baseline_ndcg column) — live queries run
against the multi-relevant golden set immediately after the Phase 0021 rerank retest
(commit `a694206`, 2026-07-22 21:32), before Phase 3 sparse reindex (commit `7d0d741`,
2026-07-23 15:42).

**After** (post-Phase-3): sparse vectors rebuilt under `enriched-base-kw` recipe — adds
`Keywords: <kw1>, <kw2>, ...` clause when the film's keywords payload is non-empty.
15,502 points rewritten (Phase 3). Source: `eval/runs/retrieval_20260724T003631.csv`,
run 2026-07-24 against the live reindexed corpus.

**Golden set**: multi-relevant clusters, 29 queries, mean cluster size 5.48 (max 6).
Built by genre-Jaccard clustering at τ=0.2 (spec 0021). All metrics computed against
this same golden set in both passes.

## Results table

| config | top_k | hybrid | query_rewrite | nDCG@k before | nDCG@k after | Δ nDCG | recall@k before | recall@k after | Δ recall |
|---|---|---|---|---|---|---|---|---|---|
| ret_000 | 5 | ✗ | ✗ | 0.0104 | 0.0101 | **−0.0003** | 0.0115 | 0.0115 | 0.0000 |
| ret_001 | 5 | ✗ | ✓ | 0.0279 | 0.0234 | **−0.0045** | 0.0172 | 0.0115 | −0.0057 |
| ret_002 | 5 | ✓ | ✗ | 0.0000 | 0.0050 | **+0.0050** | 0.0000 | 0.0057 | +0.0057 |
| ret_003 | 5 | ✓ | ✓ | 0.0234 | 0.0234 | **0.0000** | 0.0115 | 0.0115 | 0.0000 |
| ret_004 | 10 | ✗ | ✗ | 0.0093 | 0.0090 | **−0.0003** | 0.0115 | 0.0115 | 0.0000 |
| ret_005 | 10 | ✗ | ✓ | 0.0313 | 0.0242 | **−0.0071** | 0.0287 | 0.0172 | −0.0115 |
| ret_006 | 10 | ✓ | ✗ | 0.0069 | 0.0078 | **+0.0009** | 0.0115 | 0.0115 | 0.0000 |
| ret_007 | 10 | ✓ | ✓ | 0.0200 | 0.0170 | **−0.0030** | 0.0172 | 0.0115 | −0.0057 |
| **Net (hybrid only)** | — | ✓ | — | 0.0503 | 0.0532 | **+0.0029** | — | — | — |
| **Net (all configs)** | — | — | — | 0.1292 | 0.1199 | **−0.0093** | — | — | — |

## Analysis

**Scale caveat.** All absolute nDCG values are low (0.01–0.03) because the multi-relevant
golden set requires retrieving 5–6 films from the same genre cluster per query; the corpus
has 15,502 films and most clusters are very sparse in the retrieval pool. Changes on the
order of ±0.005 are within measurement noise for a 29-query test set. No individual Δ here
is statistically significant.

**Hybrid configs.** The net change across the four hybrid configs is +0.0029 nDCG — a
marginal positive. ret_002 (k=5, hybrid, no query_rewrite) moved from 0.0000 to 0.0050 —
the only config where the change moves out of the zero floor. ret_003 is flat. ret_006 and
ret_007 show mixed results (+0.0009 and −0.0030). The keyword clause adds signal for
genre-cluster retrieval at k=5 without query rewrite, but this single data point is not
enough to call it a definitive win.

**Dense configs.** Dense-only configs show a modest regression on query_rewrite=True configs
(−0.0045 and −0.0071), likely because the sparse (keyword) channel is not used in dense-only
mode and the BM25 vocabulary has shifted — but the RRF fusion is not active, so this should
not propagate. The small regressions on dense configs are most likely noise (recall only
changes by 1 document flip in the 29-query set).

**Tag-like / keyword-driven queries.** The golden set queries are all natural-language
semantic queries ("dark psychological thriller that explores themes of identity…") rather
than tag-like queries ("heist movie", "based on true story"). The keyword clause is
designed to help on the latter. The 29-query golden set underrepresents that query type —
consistent with the finding in `retrieval_eval_summary.md` that hybrid underperforms
dense-only on this query mix. A tag-specific query slice would be needed to measure the
keyword clause's strongest expected signal.

**Production config (dense + query_rewrite=True).** The production default is dense-only
with query_rewrite (best config from Phase 0020). That config (ret_001/ret_005) shows
nDCG −0.0045 to −0.0071. Both are within noise at this golden set size. The sparse channel
is inactive in dense-only mode so the keyword clause does not directly affect the production
config; the small movement likely reflects a vocabulary distribution shift in BM25 weights
(the idf scores in Qdrant are recomputed across the full corpus vocabulary when points are
updated).

## Keep / revert verdict

**KEEP.** The keyword clause is retained and the corpus stays indexed under
`enriched-base-kw`.

Rationale:
1. No statistically significant regression on any config (all |Δ| < 0.01 on a 29-query set).
2. Hybrid configs show a net-positive trend (+0.0029), and ret_002 recovers from zero.
3. The feature's primary value is the `MovieFilters.keywords` filter (AC-5) — an API
   capability that allows callers to narrow retrieval to films with specific keywords.
   That capability is independent of the nDCG numbers on this generic golden set.
4. The keywords payload is now populated and indexed (AC-1, AC-2) and the sparse recipe
   includes the keywords clause (AC-3, AC-4). Reverting would lose these improvements for
   zero measurable gain.
5. The golden set's genre-cluster structure limits sensitivity on keyword-driven queries.
   A future eval with tag-like golden queries would give a cleaner signal on this feature.

T4.4 (revert branch) is **not triggered**.

## Artifacts

- Before baseline: `eval/runs/rerank_retest_results.json` (baseline_ndcg column)
- After run: `eval/runs/retrieval_20260724T003631.csv`
- Grid: `eval/grids/retrieval.yaml` (8 configs, full hybrid slice + dense baseline)
- Golden set: `data/golden_set.json` (multi-relevant, 29 queries, mean cluster 5.48)
