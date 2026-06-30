"""Integration test: golden set builds with >= 20 queries (watchlist has 28)."""
import pytest

from eval.golden import build_golden_set


@pytest.mark.integration
def test_golden_set_builds():
    golden = build_golden_set()
    assert len(golden.queries) >= 20
    assert len(golden.holdout_tmdb_ids) >= 20
    for q in golden.queries:
        assert q.text
        assert len(q.target_tmdb_ids) >= 1
