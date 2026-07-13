"""Labelled fixture set for the looks_title_shaped(query) heuristic.

Each entry is (query, expected_bool) where True = title-shaped (the query
plausibly names a specific film and warrants the LLM fallback on 0 hits),
False = no-title / recommend-verb / discovery-intent query (LLM call must
be suppressed).

Used by AC-1.5 (zero-LLM-call budget assertion) in test_disambiguation.py:
a spy on _extract_title_via_llm must be called zero times across all
False-labelled fixtures in detect_title_collision when find_by_exact_title
returns [].
"""

from __future__ import annotations

# (query, is_title_shaped)
TITLE_SHAPED_QUERIES: list[tuple[str, bool]] = [
    # -----------------------------------------------------------------
    # EN: title-shaped (True)  — inform-intent, question about a named film
    # -----------------------------------------------------------------
    ("When was Obsession released?", True),
    ("Who directed Obsession?", True),
    ("Tell me about Inception", True),
    ("What is the theme of Parasite?", True),
    ("Where can I watch Dune?", True),
    ("What year was The Godfather made?", True),
    ("Who starred in Knives Out?", True),
    ("Obsession", True),
    ("Glass Onion", True),
    # -----------------------------------------------------------------
    # PT: title-shaped (True)  — inform-intent questions about named films
    # -----------------------------------------------------------------
    ("Quando foi lançado Obession?", True),
    ("Quem dirigiu Inception?", True),
    ("Me fale sobre Parasita", True),
    ("Onde posso assistir Duna?", True),
    ("Qual o elenco de O Poderoso Chefão?", True),
    # -----------------------------------------------------------------
    # EN: NOT title-shaped (False) — recommend/discovery intent, no named film
    # -----------------------------------------------------------------
    ("recommend something slow and tense", False),
    ("suggest me a good thriller", False),
    ("find me something scary", False),
    ("show me movies about space", False),
    ("something with a slow burn", False),
    ("films like Knives Out", False),
    ("movies like Parasite", False),
    ("similar to Arrival", False),
    ("in the style of Kubrick", False),
    ("what should I watch tonight?", False),
    ("recommend a film with a twist ending", False),
    ("I want something funny", False),
    ("suggest a good drama", False),
    # -----------------------------------------------------------------
    # PT: NOT title-shaped (False) — recommend/discovery intent in Portuguese
    # -----------------------------------------------------------------
    ("recomende algo assustador", False),
    ("sugira um bom thriller", False),
    ("me indique um filme de suspense", False),
    ("algo com ritmo lento", False),
    ("filmes como Parasita", False),
    ("parecido com Chega de Saudade", False),
    ("similar a O Auto da Compadecida", False),
    ("no estilo de Kubrick", False),
    ("o que assistir hoje à noite?", False),
    ("quero assistir algo engraçado", False),
]
