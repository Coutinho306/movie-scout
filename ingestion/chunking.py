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


def build_movie_embed_text(
    metadata: TmdbMovieMetadata, *, recipe: str = "base"
) -> str:
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
