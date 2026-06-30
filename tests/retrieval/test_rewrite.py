"""Integration test for retrieval.rewrite.rewrite_query."""

from retrieval.rewrite import rewrite_query


def test_rewrite_query_returns_string() -> None:
    result = rewrite_query("filmes tipo Stalker")
    assert isinstance(result, str)
    assert len(result) > 0


def test_rewrite_query_cached() -> None:
    q = "filme melancólico anos 70"
    r1 = rewrite_query(q)
    r2 = rewrite_query(q)
    assert r1 == r2  # same object from cache
