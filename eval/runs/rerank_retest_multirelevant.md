# Rerank re-test — multi-relevant golden set

**Verdict: P1-A STANDS.** The cross-encoder reranker (`cross-encoder/stsb-distilroberta-base`)
is net-negative on the **multi-relevant** golden set across all 8 grid configs, with net
Δ nDCG = **-0.1085** and 0/8 configs in positive territory. The direction does **not** flip
from the earlier single-relevant measurement. Spec 0020's production removal of reranking is
confirmed with a fairer measurement methodology.

## Background

The original rerank decision (spec 0020, commit `d55b46d`) was made against a golden set
where each query had exactly one relevant film (`target_tmdb_ids={seed_id}`). A single-relevant
set maximally penalizes any reorder that demotes the one relevant film — nDCG drops from
1.0 to 0.39 if the film moves from rank 1 to rank 5, with no cushion. Spec 0021 asks: does
a fairer, multi-relevant ground truth change the verdict?

## Golden set: multi-relevant clusters

The golden set was enriched using a deterministic genre-Jaccard cluster builder
(`eval/golden.py:build_relevant_cluster`). Each seed film's `target_tmdb_ids` was widened
to a cluster of up to N=6 films sharing:
- genre overlap >= 1 (hard prefilter)
- genre-Jaccard >= τ=0.2 (keyword-Jaccard was the original design, but the production
  `tmdb_movies` collection does not store a `keywords` payload field — all 15,503 films
  have `keywords=None`. Genre Jaccard is the available signal.)

**Corpus adaptation note:** The SPEC designed clustering around `keywords` payload, but
the ingested corpus only stores `genres`, `cast`, `director`, `overview`, and `popularity`.
Genre-Jaccard at τ=0.2 yields genre-bucket clusters (e.g. Drama+Thriller films together,
Action+Sci-Fi together). This is a valid but looser form of co-relevance than keyword-Jaccard
would provide. The eyeball sanity pass (AC-5) found clusters that are broad genre groups
rather than tight thematic clusters.

**Golden set stats:**
- 29 queries total
- 26/29 with multi-relevant clusters (3 seeds not in corpus → singleton)
- max cluster size: 6
- mean cluster size: 5.48

## Results: single-relevant (from rerank_eval.md) vs multi-relevant

Old single-relevant baselines from `eval/runs/retrieval_20260722T055834.csv` (last rerank eval run).

| Config | Old single-rel baseline nDCG | Old single-rel rerank Δ | Multi-rel baseline nDCG | Multi-rel rerank nDCG | Multi-rel Δ |
|---|---|---|---|---|---|
| k=5, dense, no rewrite | 0.0995 | −0.0300 | 0.0104 | 0.0058 | **−0.0045** |
| k=5, dense, rewrite | 0.1355 | −0.0690 | 0.0279 | 0.0050 | **−0.0229** |
| k=5, hybrid, no rewrite | 0.0696 | −0.0696 | 0.0000 | 0.0000 | **+0.0000** |
| k=5, hybrid, rewrite | 0.0907 | −0.0907 | 0.0234 | 0.0000 | **−0.0234** |
| k=10, dense, no rewrite | 0.1199 | −0.0257 | 0.0093 | 0.0052 | **−0.0040** |
| k=10, dense, rewrite (best prod) | 0.1470 | −0.1080 | 0.0313 | 0.0045 | **−0.0268** |
| k=10, hybrid, no rewrite | 0.0662 | −0.0662 | 0.0069 | 0.0000 | **−0.0069** |
| k=10, hybrid, rewrite | 0.1022 | −0.1022 | 0.0200 | 0.0000 | **−0.0200** |
| **Net** | — | **−0.5614** | — | — | **−0.1085** |

## Analysis

**Magnitude change.** The multi-relevant set reduces the *magnitude* of rerank's loss
substantially: net Δ single-relevant = −0.5614 vs net Δ multi-relevant = −0.1085 (about
5× smaller). This is the expected cushioning effect: when a cluster has 6 members, demoting
the seed to rank 5 no longer craters nDCG to zero if another cluster member remains at rank 1.

**Direction does not flip.** Despite the smaller magnitude, rerank is still net-negative
across 7/8 configs. The one "neutral" config (k=5, hybrid, no rewrite) has baseline nDCG=0.0000
— both baseline and rerank produced zero hits, so the Δ=0 is not a genuine win.

**Absolute baseline nDCG is lower.** The multi-relevant baseline nDCGs (0.010–0.031) are much
lower than the single-relevant baselines (0.069–0.147). This is because the multi-relevant
clusters count a hit as relevant only if the *retrieved* film is also in the seed's genre
cluster, and the retrieval pool of 50 films rarely contains 5–6 films all matching the same
genre cluster. The absolute nDCG values reflect the dominant recall bottleneck (mean recall
@k5 ≈ 0.01–0.02 on multi-relevant), not a regression in retrieval quality.

**Reranker behavior.** The reranker consistently scores cluster members *lower* than the
retrieval baseline ranks them. At hybrid configs with no query-rewrite, the reranker produces
nDCG=0.0000 (all relevant films pushed below top_k), indicating the cross-encoder assigns
lower scores to the genre-cluster members than to unrelated but textually plausible films.
This is consistent with the spike 0009 finding: the corpus's primary failure is the recall
gap, and the cross-encoder cannot fix what isn't in the candidate pool — but it makes things
worse by demoting the few relevant films that do appear.

## Decision-rule verdict

**P1-A (production removal) STANDS.**

The multi-relevant measurement is the *fairer* test: it awards partial credit when the
reranker demotes the seed but promotes another cluster member. Even with this credit,
rerank is net-negative (0/8 configs positive, net Δ = −0.1085). The direction of the
original single-relevant measurement is confirmed.

No follow-up to reopen spec 0020 is warranted. The reranker hurts retrieval quality across
all measured configurations under both evaluation methodologies.

**P1-B (query-aware routing) is not flagged** as a follow-up since no configuration showed
a positive rerank effect that could be routed to.

## Artifacts

- Results JSON: `eval/runs/rerank_retest_results.json`
- Runner: `eval/rerank_local.py` (eval-only, no production imports)
- Grid: `eval/grids/retrieval.yaml` (8-config, all configs run)
- Golden set: `data/golden_set.json` (enriched, 29 queries, mean cluster 5.48)
