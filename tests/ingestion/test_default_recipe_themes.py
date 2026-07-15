"""AC1 — themes is the production default recipe.

Pure config tests: no network, no mocked clients needed.

Verifies:
- plain Settings() → embed_text_recipe=="themes", _is_default_variant() True,
  movies_collection=="tmdb_movies", reviews_collection=="tmdb_reviews" (unsuffixed).
- RetrievalSettings().ingestion().movies_collection=="tmdb_movies" (ingest+retrieval lockstep).
- Settings(embed_text_recipe="keywords") is now non-default → suffixed collection.
- Settings.from_variant_suffix("default") round-trips to a themes-recipe Settings.
"""

from __future__ import annotations

from ingestion.config import Settings
from retrieval.config import RetrievalSettings


def test_plain_settings_embed_recipe_is_themes() -> None:
    """A bare Settings() must default to themes."""
    s = Settings()
    assert s.embed_text_recipe == "themes", (
        f"expected embed_text_recipe='themes', got {s.embed_text_recipe!r}"
    )


def test_plain_settings_is_default_variant() -> None:
    """A bare Settings() must be recognised as the default variant."""
    s = Settings()
    assert s._is_default_variant(), (
        "_is_default_variant() should be True for plain Settings()"
    )


def test_plain_settings_movies_collection_unsuffixed() -> None:
    """Default settings → unsuffixed tmdb_movies collection."""
    s = Settings()
    assert s.movies_collection == "tmdb_movies", (
        f"expected 'tmdb_movies', got {s.movies_collection!r}"
    )


def test_plain_settings_reviews_collection_unsuffixed() -> None:
    """Default settings → unsuffixed tmdb_reviews collection."""
    s = Settings()
    assert s.reviews_collection == "tmdb_reviews", (
        f"expected 'tmdb_reviews', got {s.reviews_collection!r}"
    )


def test_retrieval_ingestion_lockstep() -> None:
    """RetrievalSettings().ingestion().movies_collection must match the plain ingest default."""
    retrieval_col = RetrievalSettings().ingestion().movies_collection
    ingest_col = Settings().movies_collection
    assert retrieval_col == "tmdb_movies", (
        f"RetrievalSettings().ingestion().movies_collection = {retrieval_col!r}, want 'tmdb_movies'"
    )
    assert retrieval_col == ingest_col, (
        f"ingest ({ingest_col!r}) and retrieval ({retrieval_col!r}) collections diverged"
    )


def test_keywords_is_now_non_default() -> None:
    """Settings(embed_text_recipe='keywords') must NOT be the default variant."""
    k = Settings(embed_text_recipe="keywords")
    assert not k._is_default_variant(), (
        "keywords should be a non-default variant after themes promotion"
    )


def test_keywords_collection_has_suffix() -> None:
    """keywords recipe now produces a suffixed collection name."""
    k = Settings(embed_text_recipe="keywords")
    assert k.movies_collection.endswith("_keywords"), (
        f"keywords collection should end with '_keywords', got {k.movies_collection!r}"
    )


def test_from_variant_suffix_default_roundtrips_to_themes() -> None:
    """from_variant_suffix('default') must yield a themes-recipe Settings."""
    s = Settings.from_variant_suffix("default")
    assert s.embed_text_recipe == "themes", (
        f"from_variant_suffix('default') yielded recipe={s.embed_text_recipe!r}, want 'themes'"
    )
    assert s._is_default_variant(), (
        "from_variant_suffix('default') should produce the default variant"
    )
    assert s.movies_collection == "tmdb_movies"
