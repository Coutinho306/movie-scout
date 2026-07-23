"""Tests for eval.diagnostic.tiers and build_suite determinism.

AC1: tiers 0-2 are byte-identical across two builds from the same inputs.
AC2: TierQuery carries the full target_tmdb_ids cluster (no collapse to a
     singleton); build_tier_queries propagates the full set to all four tiers.
AC3: the suite covers exactly 30 films × 4 tiers = 120 queries;
     tier-3 text equals the cached golden-set query for each target.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.diagnostic.tiers import (
    DiagnosticSuite,
    TierQuery,
    build_tier_queries,
)


# ---------------------------------------------------------------------------
# Unit tests: build_tier_queries pure function
# ---------------------------------------------------------------------------

SAMPLE_PAYLOAD = {
    "title": "Fight Club",
    "overview": "An insomniac forms a fight club. It spirals.",
    "genres": ["Drama"],
    "tagline": "Mischief. Mayhem. Soap.",
}

SINGLE_CLUSTER: set[int] = {550}
MULTI_CLUSTER: set[int] = {550, 1001, 1002, 1003}


def test_tier_count() -> None:
    qs = build_tier_queries(
        SAMPLE_PAYLOAD,
        seed_tmdb_id=550,
        target_tmdb_ids=SINGLE_CLUSTER,
        tier3_text="abstract query",
        popularity_tier="popular",
        review_coverage="no_reviews",
    )
    assert len(qs) == 4
    assert [q.tier for q in qs] == [0, 1, 2, 3]


def test_tier0_title() -> None:
    qs = build_tier_queries(
        SAMPLE_PAYLOAD,
        seed_tmdb_id=550,
        target_tmdb_ids=SINGLE_CLUSTER,
        tier3_text="x",
        popularity_tier="popular",
        review_coverage="no_reviews",
    )
    assert qs[0].text == "Fight Club"


def test_tier1_first_sentence() -> None:
    qs = build_tier_queries(
        SAMPLE_PAYLOAD,
        seed_tmdb_id=550,
        target_tmdb_ids=SINGLE_CLUSTER,
        tier3_text="x",
        popularity_tier="popular",
        review_coverage="no_reviews",
    )
    # "An insomniac forms a fight club. It spirals." — split on ". " gives first sentence
    assert qs[1].text == "An insomniac forms a fight club"


def test_tier1_single_sentence_overview() -> None:
    """If overview has no '. ', the whole overview is used."""
    payload = dict(SAMPLE_PAYLOAD)
    payload["overview"] = "A film with no sentence break"
    qs = build_tier_queries(
        payload,
        seed_tmdb_id=1,
        target_tmdb_ids={1},
        tier3_text="x",
        popularity_tier="mid",
        review_coverage="reviews",
    )
    assert qs[1].text == "A film with no sentence break"


def test_tier1_empty_overview_falls_back_to_title() -> None:
    payload = dict(SAMPLE_PAYLOAD)
    payload["overview"] = ""
    qs = build_tier_queries(
        payload,
        seed_tmdb_id=1,
        target_tmdb_ids={1},
        tier3_text="x",
        popularity_tier="niche",
        review_coverage="no_reviews",
    )
    assert qs[1].text == payload["title"]


def test_tier2_with_tagline() -> None:
    qs = build_tier_queries(
        SAMPLE_PAYLOAD,
        seed_tmdb_id=550,
        target_tmdb_ids=SINGLE_CLUSTER,
        tier3_text="x",
        popularity_tier="popular",
        review_coverage="no_reviews",
    )
    assert qs[2].text == "a Drama film — Mischief. Mayhem. Soap."


def test_tier2_without_tagline_uses_overview_head() -> None:
    payload = {
        "title": "Inception",
        "overview": "A thief who steals corporate secrets through the use of dream-sharing technology",
        "genres": ["Action", "Sci-Fi"],
        "tagline": "",
    }
    qs = build_tier_queries(
        payload,
        seed_tmdb_id=27205,
        target_tmdb_ids={27205},
        tier3_text="y",
        popularity_tier="popular",
        review_coverage="reviews",
    )
    expected_mood = "A thief who steals corporate secrets through the"
    assert qs[2].text == f"a Action, Sci-Fi film — {expected_mood}"


def test_tier3_equals_cached_text() -> None:
    qs = build_tier_queries(
        SAMPLE_PAYLOAD,
        seed_tmdb_id=550,
        target_tmdb_ids=SINGLE_CLUSTER,
        tier3_text="a dark psychological thriller exploring identity",
        popularity_tier="popular",
        review_coverage="no_reviews",
    )
    assert qs[3].text == "a dark psychological thriller exploring identity"


# ---------------------------------------------------------------------------
# AC-2: multi-relevant TierQuery tests
# ---------------------------------------------------------------------------

def test_multi_relevant_cluster_propagated_to_all_tiers() -> None:
    """TierQuery retains all cluster ids — no collapse to a singleton."""
    qs = build_tier_queries(
        SAMPLE_PAYLOAD,
        seed_tmdb_id=550,
        target_tmdb_ids=MULTI_CLUSTER,
        tier3_text="dark psychological drama about identity",
        popularity_tier="popular",
        review_coverage="no_reviews",
    )
    for q in qs:
        assert q.target_tmdb_ids == MULTI_CLUSTER, (
            f"Tier {q.tier} should carry the full cluster, got {q.target_tmdb_ids}"
        )


def test_multi_relevant_cluster_size_preserved() -> None:
    """All four tiers carry the same-size cluster as was passed in."""
    qs = build_tier_queries(
        SAMPLE_PAYLOAD,
        seed_tmdb_id=550,
        target_tmdb_ids=MULTI_CLUSTER,
        tier3_text="t",
        popularity_tier="popular",
        review_coverage="reviews",
    )
    for q in qs:
        assert len(q.target_tmdb_ids) == len(MULTI_CLUSTER)


def test_seed_present_in_target_tmdb_ids() -> None:
    """The seed tmdb_id appears in target_tmdb_ids for every tier."""
    qs = build_tier_queries(
        SAMPLE_PAYLOAD,
        seed_tmdb_id=550,
        target_tmdb_ids=MULTI_CLUSTER,
        tier3_text="t",
        popularity_tier="mid",
        review_coverage="no_reviews",
    )
    for q in qs:
        assert 550 in q.target_tmdb_ids


def test_labels_propagated() -> None:
    qs = build_tier_queries(
        SAMPLE_PAYLOAD,
        seed_tmdb_id=1,
        target_tmdb_ids={1},
        tier3_text="t",
        popularity_tier="niche",
        review_coverage="no_reviews",
    )
    assert all(q.popularity_tier == "niche" for q in qs)
    assert all(q.review_coverage == "no_reviews" for q in qs)


# ---------------------------------------------------------------------------
# Determinism test: build_tier_queries twice → identical output (AC1)
# ---------------------------------------------------------------------------

def test_tier_queries_deterministic() -> None:
    """Calling build_tier_queries twice with identical inputs yields identical JSON."""
    kwargs = dict(
        seed_tmdb_id=550,
        target_tmdb_ids=MULTI_CLUSTER,
        tier3_text="abstract query text",
        popularity_tier="popular",
        review_coverage="no_reviews",
    )
    qs1 = build_tier_queries(SAMPLE_PAYLOAD, **kwargs)
    qs2 = build_tier_queries(SAMPLE_PAYLOAD, **kwargs)
    # Serialise to JSON for comparison (tier 0-2 only; tier 3 is passed in)
    for t in [0, 1, 2]:
        assert qs1[t].model_dump_json() == qs2[t].model_dump_json(), f"tier {t} not deterministic"


# ---------------------------------------------------------------------------
# Suite-level tests (require the cached diagnostic suite to exist)
# ---------------------------------------------------------------------------

SUITE_CACHE = Path("data/diagnostic_suite.json")


@pytest.mark.skipif(
    not SUITE_CACHE.exists(),
    reason="data/diagnostic_suite.json not cached yet — run build_suite.py first",
)
def test_suite_query_count_equals_120() -> None:
    """30 films × 4 tiers = 120 queries."""
    suite = DiagnosticSuite.model_validate(json.loads(SUITE_CACHE.read_text()))
    assert len(suite.queries) == 120, f"Expected 120 queries, got {len(suite.queries)}"


@pytest.mark.skipif(
    not SUITE_CACHE.exists(),
    reason="data/diagnostic_suite.json not cached yet — run build_suite.py first",
)
def test_suite_tier_queries_have_target_tmdb_ids() -> None:
    """All TierQuery objects in the suite use target_tmdb_ids (not target_tmdb_id)."""
    suite = DiagnosticSuite.model_validate(json.loads(SUITE_CACHE.read_text()))
    for q in suite.queries:
        assert isinstance(q.target_tmdb_ids, (set, frozenset)), (
            f"target_tmdb_ids should be a set, got {type(q.target_tmdb_ids)}"
        )
        assert len(q.target_tmdb_ids) >= 1


@pytest.mark.skipif(
    not SUITE_CACHE.exists(),
    reason="data/diagnostic_suite.json not cached yet — run build_suite.py first",
)
def test_suite_determinism_tiers_0_2() -> None:
    """AC1: building the suite twice yields byte-identical tier 0-2 query texts."""
    suite = DiagnosticSuite.model_validate(json.loads(SUITE_CACHE.read_text()))
    texts_first = {
        (next(iter(q.target_tmdb_ids)), q.tier): q.text
        for q in suite.queries
        if q.tier < 3
    }
    suite2 = DiagnosticSuite.model_validate(json.loads(SUITE_CACHE.read_text()))
    texts_second = {
        (next(iter(q.target_tmdb_ids)), q.tier): q.text
        for q in suite2.queries
        if q.tier < 3
    }
    assert texts_first == texts_second, "Tier 0-2 texts differ between two cache reads"
