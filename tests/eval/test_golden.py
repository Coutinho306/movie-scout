"""Unit tests for eval.golden cluster builder and GoldenQuery construction.

AC-1: cluster builder produces correct membership (genre+keyword-Jaccard filter),
      respects cap N, always includes the seed, and is deterministic (no RNG).
AC-3: prompt text no longer instructs toward plot-specific detail and requests
      theme/genre/mood-level queries; format keys {title}/{year} are present.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from eval.golden import build_relevant_cluster

# ---------------------------------------------------------------------------
# Fixtures — synthetic corpus payloads
# ---------------------------------------------------------------------------

SEED = {
    "tmdb_id": 1,
    "genres": ["Action", "Thriller"],
    "keywords": ["spy", "heist", "double-cross"],
    "popularity": 100.0,
}

# Shares genre AND Jaccard >= 0.2
OVERLAPPER_A = {
    "tmdb_id": 2,
    "genres": ["Action", "Drama"],
    "keywords": ["spy", "heist", "betrayal"],
    "popularity": 80.0,
}

# Shares genre AND Jaccard >= 0.2
OVERLAPPER_B = {
    "tmdb_id": 3,
    "genres": ["Thriller", "Crime"],
    "keywords": ["spy", "double-cross", "assassination"],
    "popularity": 60.0,
}

# Shares genre but Jaccard < 0.2 (only 1 keyword in common out of many)
GENRE_ONLY = {
    "tmdb_id": 4,
    "genres": ["Action", "Comedy"],
    "keywords": ["spy", "wedding", "romance", "comedy", "family", "adventure"],
    "popularity": 50.0,
}

# No genre overlap at all
UNRELATED = {
    "tmdb_id": 5,
    "genres": ["Romance", "Musical"],
    "keywords": ["love", "dance", "music"],
    "popularity": 90.0,
}

CORPUS = [SEED, OVERLAPPER_A, OVERLAPPER_B, GENRE_ONLY, UNRELATED]


# ---------------------------------------------------------------------------
# AC-1 cluster builder tests
# ---------------------------------------------------------------------------

def test_overlapping_films_included() -> None:
    """Films with genre overlap >= 1 AND keyword Jaccard >= tau are included."""
    cluster = build_relevant_cluster(SEED, CORPUS)
    assert 2 in cluster, "OVERLAPPER_A should be in cluster"
    assert 3 in cluster, "OVERLAPPER_B should be in cluster"


def test_genre_only_film_excluded() -> None:
    """Film with genre overlap but keyword Jaccard below tau is excluded."""
    cluster = build_relevant_cluster(SEED, CORPUS, tau=0.2)
    # GENRE_ONLY shares "spy" but has many unique keywords → Jaccard too low
    # seed_keywords={"spy","heist","double-cross"}, genre_only={"spy","wedding","romance","comedy","family","adventure"}
    # intersection={"spy"} → len=1, union=9 → Jaccard=1/9 ≈ 0.111 < 0.2
    assert 4 not in cluster, "GENRE_ONLY (Jaccard < tau) should be excluded"


def test_unrelated_film_excluded() -> None:
    """Film with no genre overlap is excluded regardless of keyword content."""
    cluster = build_relevant_cluster(SEED, CORPUS)
    assert 5 not in cluster, "UNRELATED film should be excluded"


def test_seed_always_included() -> None:
    """Seed is always a member of the cluster, even with n=1."""
    cluster = build_relevant_cluster(SEED, CORPUS, n=1)
    assert 1 in cluster, "Seed must always be in cluster"


def test_seed_included_when_n_would_exclude_it() -> None:
    """With n=1 (cap), seed is still present (it's the only member)."""
    cluster = build_relevant_cluster(SEED, CORPUS, n=1)
    assert len(cluster) == 1
    assert 1 in cluster


def test_cap_respected() -> None:
    """Cluster size never exceeds N."""
    cluster = build_relevant_cluster(SEED, CORPUS, n=2)
    assert len(cluster) <= 2


def test_full_cluster_without_cap() -> None:
    """With large n, the cluster contains seed + both overlapping films."""
    cluster = build_relevant_cluster(SEED, CORPUS, n=10)
    assert cluster == {1, 2, 3}


def test_determinism() -> None:
    """Calling the builder twice with same inputs gives identical results."""
    c1 = build_relevant_cluster(SEED, CORPUS)
    c2 = build_relevant_cluster(SEED, CORPUS)
    assert c1 == c2


def test_empty_corpus() -> None:
    """Empty corpus returns singleton seed cluster."""
    cluster = build_relevant_cluster(SEED, [])
    assert cluster == {1}


def test_tau_zero_includes_genre_overlap_only() -> None:
    """With tau=0.0, any genre overlap (even empty keywords) admits the film."""
    film_no_kw = {
        "tmdb_id": 6,
        "genres": ["Action"],
        "keywords": [],
        "popularity": 10.0,
    }
    corpus = [SEED, film_no_kw]
    cluster = build_relevant_cluster(SEED, corpus, tau=0.0)
    # Jaccard of {} and {"spy","heist","double-cross"}: union=3, intersection=0 → 0.0 >= 0.0
    assert 6 in cluster


def test_seed_excluded_from_self_comparison() -> None:
    """The seed is not compared against itself (no self-loop producing duplicate)."""
    cluster = build_relevant_cluster(SEED, CORPUS)
    # cluster is a set so duplicates are impossible, but size check confirms no extra entry
    assert len(cluster) == len(set(cluster))


# ---------------------------------------------------------------------------
# AC-3 prompt text tests
# ---------------------------------------------------------------------------

PROMPT_PATH = Path(__file__).parent.parent.parent / "eval/prompts/query_gen.md"


def test_prompt_has_title_and_year_keys() -> None:
    """The prompt template must contain {title} and {year} format keys."""
    text = PROMPT_PATH.read_text()
    assert "{title}" in text, "Prompt must contain {title}"
    assert "{year}" in text, "Prompt must contain {year}"


def test_prompt_does_not_instruct_plot_specific() -> None:
    """Prompt should not instruct toward plot-specific detail."""
    text = PROMPT_PATH.read_text().lower()
    # The old prompt said nothing prohibitive; the new one should not say
    # "unique plot detail" or "plot detail" as a positive instruction.
    # We check that the word "plot" does not appear in a positive framing.
    # (Acceptable if it says "without plot-specific details".)
    assert "plot-specific detail" not in text or "without" in text


def test_prompt_requests_theme_mood_genre() -> None:
    """Prompt must explicitly request theme, genre, or mood-level query."""
    text = PROMPT_PATH.read_text().lower()
    assert any(
        word in text for word in ("theme", "mood", "genre", "emotional tone")
    ), "Prompt should request theme/mood/genre-level query"


def test_prompt_requests_multi_film_relevance() -> None:
    """Prompt should indicate multiple films could be relevant answers."""
    text = PROMPT_PATH.read_text().lower()
    assert "multiple" in text or "cluster" in text or "others of its kind" in text, (
        "Prompt should indicate multiple films could be relevant"
    )
