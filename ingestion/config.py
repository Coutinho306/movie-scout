"""Shared ingestion settings — single source of truth for embedder, chunk params, and collection names."""

from __future__ import annotations

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

# Short tokens used in variant-suffixed collection names.
_EMBEDDER_TO_TOKEN: dict[str, str] = {
    "openai-3-small": "3small",
    "openai-3-large": "3large",
    "minilm": "minilm",
    "bge-small": "bgesmall",
}
_TOKEN_TO_EMBEDDER = {v: k for k, v in _EMBEDDER_TO_TOKEN.items()}

_DEFAULT_CHUNK_MAX = 300
_DEFAULT_CHUNK_OVERLAP = 50
_DEFAULT_RECIPE = "base"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    embedder: Literal["openai-3-small", "openai-3-large", "minilm", "bge-small"] = "openai-3-small"
    chunk_max_tokens: int = _DEFAULT_CHUNK_MAX
    chunk_overlap_tokens: int = _DEFAULT_CHUNK_OVERLAP
    # What text we embed per movie: "base" (title/genres/cast/tagline/overview) or
    # "keywords" (base + TMDB keywords). The pre-spike found this is the real lever.
    embed_text_recipe: Literal["base", "keywords"] = _DEFAULT_RECIPE
    # Calibration namespace: when True, collections get a "calib_" marker so a
    # sample ingest never maps onto (or clobbers) the production default collections.
    sample: bool = False

    @property
    def embedder_dim(self) -> int:
        return {
            "openai-3-small": 1536,
            "openai-3-large": 3072,
            "minilm": 384,
            "bge-small": 384,
        }[self.embedder]

    @property
    def variant_suffix(self) -> str:
        """Collision-free token encoding embedder + chunk + embed_text recipe.

        All three knobs are encoded so distinct configs never share a collection.
        Round-trips via ``from_variant_suffix``.
        """
        token = _EMBEDDER_TO_TOKEN[self.embedder]
        suffix = f"{token}_c{self.chunk_max_tokens}o{self.chunk_overlap_tokens}"
        if self.embed_text_recipe != _DEFAULT_RECIPE:
            suffix += f"_{self.embed_text_recipe}"
        if self.sample:
            suffix = f"calib_{suffix}"
        return suffix

    @classmethod
    def from_variant_suffix(cls, suffix: str) -> "Settings":
        """Reconstruct Settings from a variant_suffix token (inverse of the property).

        ``"default"`` yields plain defaults. Lets retrieval reconstruct the exact
        ingestion config a collection was built with, so the query embeds in the
        same vector space as the stored points.
        """
        if suffix == "default":
            return cls()

        parts = suffix.split("_")
        sample = False
        if parts and parts[0] == "calib":
            sample = True
            parts = parts[1:]

        token = parts[0]
        if token not in _TOKEN_TO_EMBEDDER:
            raise ValueError(f"unknown embedder token in variant {suffix!r}")
        embedder = _TOKEN_TO_EMBEDDER[token]

        chunk_max = _DEFAULT_CHUNK_MAX
        chunk_overlap = _DEFAULT_CHUNK_OVERLAP
        recipe = _DEFAULT_RECIPE
        for part in parts[1:]:
            if part.startswith("c") and "o" in part:
                c_str, o_str = part[1:].split("o", 1)
                chunk_max, chunk_overlap = int(c_str), int(o_str)
            elif part in ("base", "keywords"):
                recipe = part
            else:
                raise ValueError(f"unrecognized variant token part {part!r} in {suffix!r}")

        return cls(
            embedder=embedder,  # type: ignore[arg-type]
            chunk_max_tokens=chunk_max,
            chunk_overlap_tokens=chunk_overlap,
            embed_text_recipe=recipe,  # type: ignore[arg-type]
            sample=sample,
        )

    def _is_default_variant(self) -> bool:
        return (
            not self.sample
            and self.embedder == "openai-3-small"
            and self.chunk_max_tokens == _DEFAULT_CHUNK_MAX
            and self.chunk_overlap_tokens == _DEFAULT_CHUNK_OVERLAP
            and self.embed_text_recipe == _DEFAULT_RECIPE
        )

    @property
    def movies_collection(self) -> str:
        if self._is_default_variant():
            return "tmdb_movies"
        return f"tmdb_movies__{self.variant_suffix}"

    @property
    def reviews_collection(self) -> str:
        if self._is_default_variant():
            return "tmdb_reviews"
        return f"tmdb_reviews__{self.variant_suffix}"
