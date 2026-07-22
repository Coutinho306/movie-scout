# Re-ranking eval — model swap tried, still a net loss

**Verdict: LOSS.** Swapping the cross-encoder to a short-text-suited model
(`cross-encoder/stsb-distilroberta-base`) did not rescue reranking. It's worse
than the old `ms-marco-MiniLM-L-6-v2` on net nDCG delta, worse at the best
production config, and slower. Reranking stays **off by default**; keep/route/
remove is deferred to a follow-up spike (see Takeaway).

**Question:** does swapping to a cross-encoder trained on short, symmetric
text pairs (STS-B) — rather than long web-passage relevance (MS-MARCO) — fix
reranking's off-distribution problem on this corpus's short `title + overview`
text shape?

## Before (both code paths reconciled)

Two rerank code paths existed: production (`retrieval/movies.py`, reads
`settings.rerank`, widens fetch to `min(k*3, 30)`) and a diagnostic-only path
(`eval/diagnostic/configs.py`'s `rerank_widened`/`hyde_rerank`) that bypassed
the flag with a hardcoded 50-item pool and never combined with hybrid. These
have been reconciled: the diagnostic configs now set `settings.rerank=True`
and flow through the same `search_movies(...)` call as every other config, so
diagnostic and grid measure the *same* pipeline. Consequence: the diagnostic's
rerank pool shrank from 50 to 30 — **diagnostic rerank numbers from this run
are not directly comparable to the pre-reconciliation diagnostic run.**

## Change

`retrieval/rerank.py`'s `_DEFAULT_MODEL` changed from
`cross-encoder/ms-marco-MiniLM-L-6-v2` to
`cross-encoder/stsb-distilroberta-base` — a model trained on short
sentence-pair semantic-similarity data, the same text *shape* as
`title + overview`, as opposed to long query→passage relevance. Signature of
`cross_encode_rerank` is unchanged; this is a default-value swap only.

## Result

Re-ran the 16-config retrieval grid (`top_k` × `hybrid` × `query_rewrite` ×
`rerank`, 29 golden queries, full 15,502-film corpus) with the new model:
`eval/runs/retrieval_20260722T055834.csv`, compared against the old-model run
`eval/runs/retrieval_20260721T202044.csv` (both vs their own `rerank=False`
baselines — baselines shift marginally run-to-run from upstream
nondeterminism, e.g. query-rewrite calls):

| Config | Baseline nDCG (old run) | Baseline nDCG (new run) | Old ms-marco Δ | New stsb-distilroberta Δ | Old rerank latency (ms) | New rerank latency (ms) |
|---|---|---|---|---|---|---|
| top_k=5, dense, no rewrite | 0.0987 | 0.0995 | −0.0399 | −0.0300 | 480 | 707 |
| top_k=5, dense, rewrite | 0.1528 | 0.1355 | −0.1035 | −0.0690 | 524 | 780 |
| top_k=5, hybrid, no rewrite | 0.0562 | 0.0696 | −0.0562 | −0.0696 | 520 | 813 |
| top_k=5, hybrid, rewrite | 0.1252 | 0.0907 | −0.1119 | −0.0907 | 542 | 860 |
| top_k=10, dense, no rewrite | 0.0987 | 0.1199 | +0.0019 | −0.0257 | 696 | 1192 |
| top_k=10, dense, rewrite (best prod config) | 0.1528 | 0.1470 | −0.0870 | −0.1080 | 687 | 1377 |
| top_k=10, hybrid, no rewrite | 0.0659 | 0.0662 | −0.0207 | −0.0662 | 832 | 1487 |
| top_k=10, hybrid, rewrite | 0.1234 | 0.1022 | −0.1134 | −0.1022 | 832 | 1490 |

**Net nDCG delta across the 8 pairs:** old ms-marco = **−0.5307**; new
stsb-distilroberta = **−0.5613**. The new model is net-worse, not net-better.

**Best production config (`k10, dense, query_rewrite`):** old ms-marco delta
−0.0870; new stsb-distilroberta delta −0.1080. The new model loses more at
the config that matters most.

**Latency:** the new model is slower across every pair (roughly 1.4–2.2×
old-model rerank latency, e.g. 687ms → 1377ms at `k10, dense, rewrite`) —
likely due to `distilroberta`'s tokenizer/embedding path being heavier under
this run's load than the smaller MiniLM backbone, on top of rerank's existing
cost. No configuration lets the new model both win on quality and cost less.

## Takeaway

The model-swap hypothesis — that reranking hurt because MS-MARCO's
long-passage training was off-distribution for short title+overview text —
does **not** hold up empirically. Swapping to an STS-B short-text cross-encoder
made net nDCG *worse*, not better, and materially slower. This is consistent
with spike 0009's second root cause: the corpus's dominant failure is a recall
gap (~0.14 @k5) that no reranker, regardless of training distribution, can fix
by reordering a pool that often doesn't contain the right films at all.

Reranking stays **off by default**. The keep/route/remove decision (P1-A:
remove reranking entirely, or P1-B: query-aware routing like the `hybrid`
flag already uses) is **deferred to a follow-up spike, out of scope here** —
this SPEC's job was only to test the model-choice hypothesis and report the
number honestly, and the number says the hypothesis was wrong.

Full run artifacts: `eval/runs/retrieval_20260722T055834.csv` (new model),
`eval/runs/retrieval_20260721T202044.csv` (old model, same baselines).
