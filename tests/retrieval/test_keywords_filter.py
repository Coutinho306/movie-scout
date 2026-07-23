"""Unit tests for MovieFilters.keywords + _build_filter wiring (T1.5/T1.6).

Phase 1, spec 0022-keywords-payload-bm25-and-filter (AC-5).

Assertions:
- MovieFilters(keywords=["heist"]) produces a Filter whose must conditions
  include a keywords MatchAny.
- MovieFilters() (no keywords) produces no keywords condition.
- Keywords condition coexists correctly with genres/cast conditions.
"""

from __future__ import annotations

from retrieval.models import MovieFilters


class TestBuildFilterKeywords:
    def test_keywords_match_any_present_when_set(self) -> None:
        """_build_filter adds a keywords MatchAny condition when keywords is set."""
        from qdrant_client.models import FieldCondition, MatchAny

        from retrieval.movies import _build_filter

        f = _build_filter(MovieFilters(keywords=["heist"]))
        assert f is not None
        assert f.must is not None

        kw_conditions = [
            c for c in f.must
            if isinstance(c, FieldCondition) and c.key == "keywords"
        ]
        assert len(kw_conditions) == 1
        cond = kw_conditions[0]
        assert isinstance(cond.match, MatchAny)
        assert "heist" in cond.match.any

    def test_keywords_match_any_multi_token(self) -> None:
        """_build_filter supports multiple keyword tokens via MatchAny."""
        from qdrant_client.models import FieldCondition

        from retrieval.movies import _build_filter

        f = _build_filter(MovieFilters(keywords=["heist", "time travel"]))
        assert f is not None
        kw_conditions = [
            c for c in f.must
            if isinstance(c, FieldCondition) and c.key == "keywords"
        ]
        assert len(kw_conditions) == 1
        assert set(kw_conditions[0].match.any) == {"heist", "time travel"}

    def test_no_keywords_condition_when_keywords_none(self) -> None:
        """_build_filter must not add a keywords condition when MovieFilters() has no keywords."""
        from qdrant_client.models import FieldCondition

        from retrieval.movies import _build_filter

        # Default MovieFilters — no keywords.
        f = _build_filter(MovieFilters())
        if f is None:
            return  # no conditions at all — fine
        kw_conditions = [
            c for c in (f.must or [])
            if isinstance(c, FieldCondition) and c.key == "keywords"
        ]
        assert len(kw_conditions) == 0

    def test_no_keywords_condition_returns_none_filter(self) -> None:
        """_build_filter returns None when MovieFilters is entirely empty."""
        from retrieval.movies import _build_filter

        f = _build_filter(MovieFilters())
        assert f is None

    def test_keywords_condition_coexists_with_genres_and_cast(self) -> None:
        """keywords, genres, and cast conditions all appear together in must."""
        from qdrant_client.models import FieldCondition

        from retrieval.movies import _build_filter

        f = _build_filter(
            MovieFilters(keywords=["heist"], genres=["Crime"], cast=["Al Pacino"])
        )
        assert f is not None
        keys = [c.key for c in f.must if isinstance(c, FieldCondition)]
        assert "keywords" in keys
        assert "genres" in keys
        assert "cast" in keys

    def test_keywords_field_is_optional_on_movie_filters(self) -> None:
        """MovieFilters.keywords defaults to None and accepts a list."""
        m = MovieFilters()
        assert m.keywords is None

        m2 = MovieFilters(keywords=["based on true story"])
        assert m2.keywords == ["based on true story"]
