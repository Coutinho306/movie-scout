# Hybrid search eval — BM25 sparse enrichment

**Question:** does enriching the BM25 sparse index (beyond overview+tagline)
improve retrieval, and where?

## Before

The sparse (lexical/BM25) vector was built from `overview + tagline` only.
Structured fields — genre, cast, director — were embedded in the dense
vector but never indexed lexically. A pure keyword search for an actor name
or a genre word returned nothing.

## Change

Sparse text now uses the same recipe as the dense `base` text: title, year,
genres, director, cast (top-5), tagline, overview. Backfilled across the
full 15,503-point movie corpus, version-tagged (`sparse_recipe:
"enriched-base"`) so the backfill is idempotent on re-run. Dense vectors
were verified untouched (element-wise) before/after.

## Result

Ran the 120-query diagnostic suite (4 difficulty tiers) comparing
`routed_hybrid` (dense + BM25, intent-routed) against `baseline_dense`
(dense only):

| Tier | baseline_dense (nDCG@10) | routed_hybrid (nDCG@10) | Δ |
|---|---|---|---|
| 0 — exact title | 0.963 | 0.963 | 0.000 |
| 1 — overview/plot | 0.854 | 0.963 | **+0.109** |
| 2 — genre/mood | 0.527 | 0.923 | **+0.396** |
| 3 — abstract | 0.148 | 0.148 | 0.000 |

Lexical recovery confirmed directly: pure-BM25 queries for an actor name
("Darmon") and a genre word ("Thriller") now return correct hits — both
returned nothing under the old recipe.

## Takeaway

Structured-field enrichment closes a real gap: queries naming a genre or
person now hit the sparse index directly instead of relying solely on dense
semantic recall, which was comparatively weak there (tier 2: 0.527 → 0.923).
No regression on exact-title matches (tier 0) or open-ended abstract queries
(tier 3, an inherent ceiling — not solved by this change).

Full run artifacts: `eval/runs/diagnostic_20260709T163235.csv`,
`_summary.txt`, `_results.json`.
