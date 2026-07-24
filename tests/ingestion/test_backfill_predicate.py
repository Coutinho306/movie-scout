"""Unit tests for the version-tag skip predicate in backfill_bm25_sparse.py (T2.5).

These are pure-logic tests — no Qdrant connection required.
"""

from __future__ import annotations

from scripts.backfill_bm25_sparse import _SPARSE_TEXT_RECIPE, _already_tagged_current_recipe


def test_untagged_point_is_not_skipped() -> None:
    """A point with no sparse_recipe tag is NOT skipped (will be rewritten)."""
    payload: dict = {"title": "Fight Club", "overview": "An insomniac forms a fight club."}
    assert not _already_tagged_current_recipe(payload)


def test_old_tagged_point_is_not_skipped() -> None:
    """A point tagged with a different/older recipe version is NOT skipped."""
    payload: dict = {"sparse_recipe": "overview + tagline"}
    assert not _already_tagged_current_recipe(payload)


def test_current_tagged_point_is_skipped() -> None:
    """A point already tagged with the current recipe IS skipped (idempotent re-run)."""
    payload: dict = {"sparse_recipe": _SPARSE_TEXT_RECIPE}
    assert _already_tagged_current_recipe(payload)


def test_current_recipe_constant_is_enriched_base_kw() -> None:
    """The recipe constant must be 'enriched-base-kw' after Phase 3 keywords bump (AC-3/AC-4)."""
    assert _SPARSE_TEXT_RECIPE == "enriched-base-kw"
