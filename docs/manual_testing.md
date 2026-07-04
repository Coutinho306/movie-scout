# Manual testing & error analysis

Automated eval (nDCG@10 golden-set grids, see `eval/runs/`) measures retrieval
quality in isolation. It doesn't catch agent-orchestration bugs — wrong tool
choice, missing guardrails, misrouted intent — that only show up when a real
person talks to the app. This doc tracks that second layer of testing: manual
`/ask` sessions, the bugs they found, and what shipped as a result.

## Round 1 — seed-film handling

| # | Query | Problem found |
|---|---|---|
| 1 | "a film like Arrival" | Arrival recommended itself alongside genuinely good picks (Ex Machina, Interstellar) |
| 2 | "a film with the same theme as Glass Onion" | Knives Out — the obvious answer — never appeared; top results were literal word-matches on "Glass" |
| 3 | "who is the director of Dune?" | Answered with a list of Dune movies instead of naming the director |

![Director of Arrival — factual question answered as a recommendation list](screenshots/Director%20of%20Arrival.png)

**Root causes found:**
- No self-exclusion when a query names a seed film.
- The agent had no way to resolve a named film ("Glass Onion") to its catalog
  entry before searching — it embedded the raw sentence and got dense-search
  lexical noise (see [`retrieval_quality.md`](retrieval_quality.md) for the
  deeper investigation this connects to).
- The orchestrator's intent classifier only recognized "what is X about"
  questions, not attribute questions ("who directed X").

**Fixed:** seed-film resolution via a deterministic (regex + TMDB lookup,
no LLM call) `search_movies_tool` path that searches using the seed's own
stored vector instead of re-embedding the query text, with the seed always
excluded from results. Intent classification broadened to cover attribute
questions. Verified fix: "movies like Glass Onion" now returns Knives Out at
rank 1, byte-identical across repeat runs.

![films like Glass Onion — after the fix, Knives Out ranks first](screenshots/Films%20like%20Glass%20Onion.png)

## Round 2 — confirming the fix, scoping what's left

Re-ran the round-1 failure patterns plus new phrasings (factual vs.
recommendation, interleaved) to confirm the fix generalized and to isolate
what was still a real gap vs. what was fixed. Result: self-exclusion and
seed-resolution held up; factual routing worked for more attribute-question
phrasings. Flagged that "where can I watch X" wasn't covered by any tool
trigger yet (see round 3).

## Round 3 — new query patterns, 4 more bug classes found

Broader manual pass: streaming availability, cast-based search, title
collisions, response verbosity.

| # | Query | Problem found |
|---|---|---|
| 5 | "Films with Ryan Gosling" | Corpus has 26 Gosling films; only 4 surfaced |
| 6 | "Films with the actor Keanu Reeves" | Corpus has 58 Reeves films; only 5 surfaced |
| 8 | "Where can I watch Project Hail Mary today?" | Returned a plot summary instead of streaming info |
| 9 | "What is the theme of Obsession? And when was it released?" | Silently answered about the wrong "Obsession" — 4 different films share that exact title in the corpus (1943 / 1976 / 2015 / 2026) |

![Films with Ryan Gosling — before the fix, only a handful of 26 surfaced](screenshots/Films%20with%20Ryan%20Gosling.png)
![Films with the actor Keanu Reeves — before the fix, only a handful of 58 surfaced](screenshots/Films%20with%20the%20actor%20Keanu%20Reeves.png)
![Where can I watch Project Hail Mary today — plot summary instead of streaming info](screenshots/Where%20can%20i%20watch%20Project%20Hail%20mary%20today.png)
![What is the theme of Obsession — silently answered about the wrong film](screenshots/What%20is%20the%20theme%20of%20Obssession%20And%20When%20it%20was%20released.png)
![Obsession 2026 — pinning the year resolves correctly, no regression](screenshots/Obsession%202026.png)

**Root causes found and fixed:**

- **Cast search used dense semantic similarity, not an exact filter.**
  "Films with actor X" is a structured lookup, not a similarity search — it
  needs an exact match against the cast field, not a ranked guess. Added a
  `cast` filter to the retrieval layer plus a deterministic query-routing
  step; confirmed against the live corpus: **26/26 Ryan Gosling, 57/58 Keanu
  Reeves** films recovered (up from 4 and 5).
- **Streaming-availability questions had no tool trigger.** The
  `tmdb_lookup_providers` tool already existed and worked — it just never
  fired for this phrasing, and its result never reached the answer synthesis
  step (it stayed inside the retrieval agent's internal message history).
  Wired the result through; verified: "is Dune on Netflix" now answers
  correctly with the actual provider list.
- **Title collisions were resolved silently to an arbitrary match.** Four
  films are named exactly "Obsession" in this corpus. The system picked one
  without asking. Added a deterministic exact-title lookup (no embedding, no
  ranking — a direct structured query) that guarantees the full set of
  same-titled films reaches the answer step, so the app now asks which one
  you meant instead of guessing. Verified: "What is the theme of Obsession?"
  → *"There are 4 films called Obsession: from 1943, 1976, 2015, and 2026 —
  which one did you mean?"* — consistent across repeat runs.
- **Responses to narrow questions were overly verbose.** "Who directed Dune?"
  returned a full paragraph with unrequested plot detail. Tightened the
  answer-synthesis prompt to match response length to question specificity —
  a one-attribute question now gets one sentence.

## Additional spot checks

More manual queries run alongside the rounds above, exercising the same
seed-film resolution and thematic-search paths on different titles:

![Films like Prestige](screenshots/Films%20like%20prestige.png)
![Films like Project Hail Mary](screenshots/Films%20like%20project%20hail%20mary.png)
![Films with the same theme as Dune](screenshots/Films%20with%20the%20same%20theme%20as%20dune.png)
![Top 5 Dark Mystery films since 2000s](screenshots/Top%205%20Dark%20Mystery%20Films%20since%202000s.png)

## What manual testing caught that automated eval didn't

The nDCG grids measure whether the retriever ranks the *correct document*
highly for a *given query* — a well-defined, gradeable metric. None of the
four round-3 bug classes were retrieval-ranking problems at all: they were
missing product logic (self-exclusion, exact-match routing, disambiguation,
response shaping) that a numeric retrieval metric structurally cannot see.
This is the practical case for treating manual/error-driven testing as a
first-class QA step, not an afterthought — a system can look fine on paper
and still be visibly wrong to a real user typing a real question.
