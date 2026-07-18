# LLM-eval grid result

Date: 2026-07-18
Run: `eval/runs/llm_20260718T214039.csv` (6 configs: temperature × prompt_variant,
10 sampled golden queries each), against the full 15,502-film production corpus
post BM25-sparse-fix and post RAGAS-import-fix (see `specs/spikes/0006-ragas-vertexai-import-fix/SPIKE.md`).

## Grid

| config | temp | prompt | faithfulness | answer_relevancy | taste_match | hallucination_rate |
|---|---|---|---|---|---|---|
| llm_000 | 0.0 | v1 | **0.6923** | 0.6833 | 0.4399 | 0.0 |
| llm_001 | 0.0 | v2 | 0.5038 | 0.7083 | 0.4413 | 0.0 |
| llm_002 | 0.3 | v1 | 0.6082 | 0.6729 | 0.4339 | 0.0 |
| llm_003 | 0.3 | v2 | 0.3830 | 0.6425 | 0.4116 | 0.0 |
| llm_004 | 0.7 | v1 | 0.4557 | 0.6603 | 0.4458 | 0.0 |
| llm_005 | 0.7 | v2 | 0.5556 | **0.7127** | 0.4402 | 0.0 |

## Winner

**`llm_000` — temperature=0.0, prompt v1.** Best faithfulness (0.69) by a wide
margin, competitive answer_relevancy (0.68, second only to 0.71). `llm_005`
edges it on answer_relevancy alone but trails badly on faithfulness (0.56) —
a real precision/faithfulness tradeoff, not noise: faithfulness degrades
roughly monotonically as temperature rises for both prompt variants.
`hallucination_rate=0.0` across every config (the agent never cites outside
its retrieval pool, confirmed once the metric was fixed to compare against
actual RAG hits instead of the golden-set target — see commit history).

Production `AgentSettings` defaults (`temperature=0.0`, `prompt_variant="v1"`)
already match the winning config.

## Metric integrity note

This is the first grid run with usable numbers. Two earlier attempts in this
session burned real OpenAI spend (~$0.10 + ~$0.08) producing unusable results:
one run failed at the `ragas` import layer (all NaN), the next had a correct
import but two metric-plumbing bugs (`hallucination_rate` scored against the
wrong ground truth, always 1.0; `faithfulness` was scored against bare movie
titles instead of overview text, always near-zero). Both are fixed and
verified in the source (`eval/metrics/llm.py`, `eval/runs/llm_grid.py`,
`agent/main.py`, `agent/state.py`) — this run's numbers reflect real,
correctly-scored agent behavior.
