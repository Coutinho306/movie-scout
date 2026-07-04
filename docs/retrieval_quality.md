# Retrieval quality investigation

This documents an open, honestly-reported problem: dense retrieval quality on
abstract, thematic queries is well below target on the full production corpus,
despite performing well in small-sample calibration. Several fixes were tried;
none closed the gap. This is reported as-is rather than hidden, because the
investigation and the ruled-out hypotheses are themselves the useful result.

## The problem

Calibration on a small sample (329 films: golden targets + distractors) picked
`text-embedding-3-small` + a keyword-augmented embedding recipe at
nDCG@10 = 0.586. The identical configuration against the full production
corpus (15,503 movies) scores **nDCG@10 = 0.12–0.15** on realistic abstract
queries ("a dark psychological thriller exploring identity and societal
norms") — well under the project's 0.45 target.

## Root cause (proven, not guessed)

An abstract-query ↔ concrete-document vocabulary mismatch. Users describe
intent thematically; the stored movie documents describe films the way TMDB
does — plot facts, cast, terse keywords. The embedding model rewards surface
word overlap over latent meaning at this vocabulary density: literal
"Dark"-titled B-movies outrank a thematically correct answer like *Fight
Club*, whose overview never uses the words "dark" or "psychological."

**Direct proof:** the exact same stored vector for a target film jumps from
unranked to rank 1 the moment the query is rephrased in concrete,
overview-style language instead of abstract thematic language. The vector
itself is correct (verified by cosine similarity against a fresh re-embed);
the retrieval index and corpus composition were also directly ruled out as
causes. A tiered diagnostic suite (four difficulty levels, from literal title
lookup to full abstract queries, deterministic — no LLM-generated
randomness) confirmed retrieval is near-perfect on concrete queries and
degrades specifically as queries become more abstract.

## What was tried

1. **HyDE query expansion** — ask an LLM to rewrite the abstract query into a
   concrete hypothetical document before embedding it. Result: made things
   *worse* on the harder query tiers, not better.
2. **Cross-encoder reranking** — widen the retrieval pool and re-rank with a
   relevance model. Result: capped out low — at full corpus scale, ~41% of
   correct answers aren't even in the widened candidate pool, so reranking
   has nothing to promote.
3. **LLM-synthesized thematic embedding recipe** — append an LLM-written
   abstract description (identity, isolation, moral ambiguity, etc.) to each
   film's embedding text, tested in two prompt iterations on a
   density-matched sample. Both underperformed the existing recipe.
4. **Wikipedia plot summaries as an alternative document source** —
   investigated, not built: Wikipedia's plot sections are *equally concrete*
   as TMDB's, just longer, so they don't address the actual mismatch.

None of these closed the gap. Each is a legitimate, evidence-based fix
attempt for the diagnosed root cause — the honest finding is that fixing this
well likely needs a genuinely different architecture (hybrid lexical +
semantic retrieval is the most promising untried direction) or a
corpus-scale-realistic target, not another prompt or embedding-recipe
variant.

## What this did NOT turn out to be

Manual end-to-end testing of the full agent (not just the isolated
retriever) surfaced a separate, more tractable set of bugs — missing
self-exclusion, no exact cast-filter, silent title-collision resolution,
missing tool routing — that were *not* retrieval-ranking problems and were
fixable directly. See [`manual_testing.md`](manual_testing.md). Fixing those
measurably improved real answer quality independent of the open nDCG problem
above, which is why both are documented separately rather than conflated.

## Current status

Open. The production corpus, embeddings, and current recipe are unchanged
from calibration's chosen configuration — no fix here has been shipped,
because none demonstrated a real improvement. The next candidate direction
(not yet attempted) is hybrid retrieval (BM25/sparse + dense fusion), which
directly targets a vocabulary-mismatch failure mode this project's own
evidence points to.
