"""Text chunking strategies for ingestion sources."""

from ingestion.models import TmdbMovieMetadata

_WORDS_PER_TOKEN = 1 / 1.3  # 1 word ≈ 1.3 tokens


def chunk_review(
    text: str,
    *,
    max_tokens: int = 300,
    overlap_tokens: int = 50,
) -> list[str]:
    words = text.split()
    max_words = int(max_tokens * _WORDS_PER_TOKEN)
    overlap_words = int(overlap_tokens * _WORDS_PER_TOKEN)
    step = max_words - overlap_words

    if len(words) <= max_words:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += step
    return chunks


def _build_keywords_text(metadata: TmdbMovieMetadata) -> str:
    """Return the keywords-recipe embed text (factored for reuse by themes recipe)."""
    genres = ", ".join(metadata.genres)
    cast = ", ".join(metadata.cast[:5])  # top 5
    text = (
        f"{metadata.title} ({metadata.year}). "
        f"Genres: {genres}. "
        f"Director: {metadata.director}. "
        f"Cast: {cast}. "
        f"{metadata.tagline}. "
        f"{metadata.overview}"
    )
    if metadata.keywords:
        # TMDB keywords name themes/motifs the overview rarely states — the lever
        # the pre-spike identified for thematic queries.
        text += f" Keywords: {', '.join(metadata.keywords)}."
    return text


def build_movie_embed_text(
    metadata: TmdbMovieMetadata, *, recipe: str = "base"
) -> str:
    if recipe == "themes":
        # themes = full keywords-recipe text + appended abstract thematic sentences.
        # Import here (not at module top) to keep chunking.py free of network I/O
        # and to make the LLM call injectable in tests.
        from ingestion.theme_extraction import extract_themes

        text = _build_keywords_text(metadata)
        themes = extract_themes(metadata)
        if themes:
            text += f" Themes: {themes}."
        return text

    genres = ", ".join(metadata.genres)
    cast = ", ".join(metadata.cast[:5])  # top 5
    text = (
        f"{metadata.title} ({metadata.year}). "
        f"Genres: {genres}. "
        f"Director: {metadata.director}. "
        f"Cast: {cast}. "
        f"{metadata.tagline}. "
        f"{metadata.overview}"
    )
    if recipe == "keywords" and metadata.keywords:
        # TMDB keywords name themes/motifs the overview rarely states — the lever
        # the pre-spike identified for thematic queries.
        text += f" Keywords: {', '.join(metadata.keywords)}."
    return text


def build_sparse_text(
    *,
    title: str,
    year: int,
    genres: list[str],
    director: str,
    cast: list[str],
    tagline: str,
    overview: str,
    keywords: list[str] | None = None,
) -> str:
    """Return the enriched-base sparse text (drift-guard canonical source).

    Emits title, year, genres, director, cast (top-5), tagline, overview, and
    — when ``keywords`` is non-empty — a ``Keywords: ...`` clause mirroring
    ``_build_keywords_text`` so dense and sparse keyword handling do not diverge.
    Default ``None`` keeps existing callers valid.

    Both the fresh-ingest sparse ``Document`` and the sparse backfill script
    call this function so the recipes cannot drift.

    Accepts flat fields (not a ``TmdbMovieMetadata``) so the backfill script
    can call it directly from the Qdrant payload without constructing a model
    instance.
    """
    genres_str = ", ".join(genres)
    cast_str = ", ".join(cast[:5])  # top-5, matching the dense base recipe
    text = (
        f"{title} ({year}). "
        f"Genres: {genres_str}. "
        f"Director: {director}. "
        f"Cast: {cast_str}. "
        f"{tagline}. "
        f"{overview}"
    )
    if keywords:
        text += f" Keywords: {', '.join(keywords)}."
    return text
