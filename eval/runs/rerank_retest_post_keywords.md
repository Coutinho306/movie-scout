# Cross-encoder rerank re-test — post-keyword-reindex verdict

**Spec**: 0024 (rerank-retest-post-keywords)
**Run date**: 2026-07-23
**Reindex boundary commit**: `7d0d741` (2026-07-23 15:42) — `_SPARSE_TEXT_RECIPE` bumped to `enriched-base-kw`
**Corpus**: 15,502 films, sparse vectors include `Keywords: <kw1>, <kw2>, ...` clause
**Runner**: `uv run python3 -m eval.rerank_local` (unmodified; cross-encoder confined to `eval/`)
**Grid coverage**: Full 8-config grid — `top_k ∈ {5, 10}` × `hybrid ∈ {false, true}` × `query_rewrite ∈ {false, true}`

---

## Corpus state

**Before** (pre-reindex): sparse vectors built from `enriched-base` recipe — overview, genres, cast,
director; no keywords. Source: `git show a694206:eval/runs/rerank_retest_results.json` (commit
`a694206`, 2026-07-22 21:32).

**After** (post-reindex): sparse vectors rebuilt under `enriched-base-kw` — adds `Keywords:` clause
when a film has non-empty keywords payload. All 15,502 points rewritten (commit `7d0d741`,
2026-07-23 15:42). Source: `eval/runs/rerank_retest_results.json` (this run, 2026-07-23, post-reindex).

**Golden set**: multi-relevant clusters, 29 queries, mean cluster size 5.48 (max 6), built by
genre-Jaccard clustering at τ=0.2 (spec 0021). No tier labels — this set cannot reproduce the
tier2/tier3 signal that drove the 0020 removal (see Caveats).

---

## Results table — post-reindex run (after)

| config_id | top_k | hybrid | query_rewrite | baseline_ndcg | rerank_ndcg | delta_ndcg | baseline_recall | rerank_recall |
|---|---|---|---|---|---|---|---|---|
| ret_000 | 5 | ✗ | ✗ | 0.0101 | 0.0074 | **-0.0027** | 0.0115 | 0.0057 |
| ret_001 | 5 | ✗ | ✓ | 0.0234 | 0.0045 | **-0.0189** | 0.0115 | 0.0057 |
| ret_002 | 5 | ✓ | ✗ | 0.0050 | 0.0000 | **-0.0050** | 0.0057 | 0.0000 |
| ret_003 | 5 | ✓ | ✓ | 0.0191 | 0.0000 | **-0.0191** | 0.0115 | 0.0000 |
| ret_004 | 10 | ✗ | ✗ | 0.0090 | 0.0066 | **-0.0024** | 0.0115 | 0.0057 |
| ret_005 | 10 | ✗ | ✓ | 0.0209 | 0.0071 | **-0.0138** | 0.0115 | 0.0115 |
| ret_006 | 10 | ✓ | ✗ | 0.0078 | 0.0033 | **-0.0045** | 0.0115 | 0.0057 |
| ret_007 | 10 | ✓ | ✓ | 0.0209 | 0.0000 | **-0.0209** | 0.0115 | 0.0000 |
| **Net (all 8)** | — | — | — | 0.1162 | 0.0289 | **-0.0873** | — | — |

Positive-delta configs: **0 / 8**

---

## Before/after comparison (pre-reindex vs post-reindex)

| config_id | Δ nDCG BEFORE (a694206) | Δ nDCG AFTER (this run) | change |
|---|---|---|---|
| ret_000 | -0.0045 | -0.0027 | +0.0018 (less negative) |
| ret_001 | -0.0229 | -0.0189 | +0.0040 (less negative) |
| ret_002 | 0.0000 | -0.0050 | -0.0050 (newly negative) |
| ret_003 | -0.0234 | -0.0191 | +0.0043 (less negative) |
| ret_004 | -0.0040 | -0.0024 | +0.0016 (less negative) |
| ret_005 | -0.0268 | -0.0138 | +0.0130 (less negative) |
| ret_006 | -0.0069 | -0.0045 | +0.0024 (less negative) |
| ret_007 | -0.0200 | -0.0209 | -0.0009 (slightly more negative) |
| **Net** | **-0.1085** | **-0.0873** | **+0.0212 (improvement)** |

The keyword reindex moved the aggregate rerank delta from -0.1085 to -0.0873, a net improvement
of +0.0212. However, the result is still firmly negative across all 8 configs with 0/8 positive.

---

## Verdict

**Significance bar**: The ~±0.005 noise floor cited in `keywords_sparse_eval.md` for a 29-query set
applies here. A result is considered positive only if: (a) net Δ nDCG exceeds the noise floor in a
positive direction, and (b) positive-config count is at least 4/8 (majority). Neither condition is
met.

The net Δ nDCG after reindex is **-0.0873** — deeply negative, far outside the ±0.005 noise floor,
and in the wrong direction. The positive-config count is **0/8**. The keyword reindex did produce
some partial improvement (net Δ improved by +0.0212 vs the pre-reindex result), but rerank remains
uniformly harmful on this golden set. Every config loses between -0.0024 and -0.0209 nDCG when
the cross-encoder reranks a pool of 50 candidates down to top_k.

**Interpretation**: The cross-encoder (`cross-encoder/stsb-distilroberta-base`) is scoring
`"{title} {overview}"` pairs against queries. The multi-relevant genre-cluster golden set has
near-zero absolute nDCG (0.00–0.02) because each cluster has 5–6 films that all look equally
relevant to the cross-encoder's passage-level scoring — but only a few appear in the widened pool
of 50. Reranking within the pool cannot surface documents that weren't fetched. The keyword-boosted
BM25 channel changes which 50 candidates enter the rerank window (hence the partial improvement),
but the cross-encoder's reordering within that window remains net-negative.

**Caveats**:

1. **No tier labels.** This golden set carries no tier labels and cannot reproduce the tier2/tier3
   signal that drove the spec 0020 removal. The 0020 verdict was based on the tiered diagnostic
   (`eval/diagnostic/`), which showed rerank going flat-to-negative on tier2 genre_mood
   (Δrecall10 = -0.2667) and tier3 abstract queries. This re-test uses the multi-relevant
   genre-cluster set (spec 0021), which is a different harness measuring a different thing.

2. **Tag-like query underrepresentation.** The 29-query set consists of natural-language semantic
   queries. The keyword clause's primary benefit is tag-like queries ("heist", "based on true
   story"). That query type is underrepresented — a flat or mildly negative result here cannot
   rule out that rerank would lift on a tag-like query slice. However, a result this uniformly and
   substantially negative (-0.0873 net, 0/8 positive) cannot be explained away by tag-like coverage
   alone.

3. **WidenedPool = 50.** The reranker fetches 50 candidates then slices to top_k. For a 29-query
   set with cluster size ~5, most relevant documents are outside the top-50 for most queries,
   limiting headroom for rerank to help.

---

## Go / no-go

**Flat/negative result confirmed**: rerank stays removed, Approach C (do nothing) is the
evidence-backed outcome, no further work on cross-encoder rerank is warranted. The keyword-boosted
pool did not flip the verdict. If a tag-aware golden slice is built in a future spec and shows
meaningful rerank lift on tag-like queries specifically, that would reopen the question — but this
re-test does not license it.

---

## Artifacts

- Before baseline: `git show a694206:eval/runs/rerank_retest_results.json`
- After run: `eval/runs/rerank_retest_results.json` (overwritten, post-reindex, 2026-07-23)
- Grid: `eval/grids/retrieval.yaml` (8 configs, full hybrid + dense slice)
- Golden set: `data/golden_set.json` (multi-relevant, 29 queries, mean cluster 5.48)
- Runner: `eval/rerank_local.py` (unmodified; cross-encoder stays eval-only)
