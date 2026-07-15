"""Run the ingestion pipeline against a variant (non-default) Qdrant collection.

Invoke via:
    uv run python3 -m ingestion.scripts.run_experiment --embedder minilm

Variant collections are auto-named (e.g. tmdb_movies__minilm_c300o50).
The default pipeline (ingestion.pipeline) always targets plain collections.
"""

import argparse
import logging
import os

from dotenv import load_dotenv

from ingestion.config import Settings
from ingestion.pipeline import drop_variant, get_qdrant_client, run_pipeline

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(message)s")
_logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ingestion pipeline against a variant Qdrant collection"
    )
    parser.add_argument(
        "--embedder",
        choices=["openai-3-small", "openai-3-large", "minilm", "bge-small"],
        default=None,
        help="embedding model variant",
    )
    parser.add_argument(
        "--chunk-max-tokens",
        type=int,
        default=None,
        help="max tokens per review chunk",
    )
    parser.add_argument(
        "--chunk-overlap-tokens",
        type=int,
        default=None,
        help="token overlap between review chunks",
    )
    parser.add_argument(
        "--embed-text-recipe",
        choices=["base", "keywords", "themes"],
        default=None,
        help="movie embed_text recipe: base, keywords, or themes (keywords + LLM abstract sentences)",
    )
    parser.add_argument(
        "--golden-sample",
        action="store_true",
        help="ingest only the fixed calibration sample (golden targets + distractors)",
    )
    parser.add_argument(
        "--tmdb-ids",
        type=str,
        default=None,
        help="comma-separated tmdb ids to ingest (overrides discovery); e.g. 550,680",
    )
    parser.add_argument(
        "--corpus",
        action="store_true",
        help=(
            "ingest an explicit id list into the production (default, unsuffixed) "
            "collections; reads data/corpus_sample.json unless --tmdb-ids is given; "
            "sets sample=False — no calib_ namespace"
        ),
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="drop and recreate this variant's Qdrant collections before loading",
    )
    parser.add_argument(
        "--drop-variant",
        action="store_true",
        help="delete this variant's Qdrant collections and exit",
    )
    parser.add_argument(
        "--refresh-taste",
        action="store_true",
        help="recompute taste_profile.json even if it already exists",
    )
    parser.add_argument(
        "--skip-taste",
        action="store_true",
        help="reuse the existing taste_profile.json without recomputing",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "skip films (movies) / films-with-chunks (reviews) already present in "
            "the target collection; default off so --rebuild stays a clean-slate op"
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="number of parallel workers for the concurrent movie loop (default: 8)",
    )
    parser.add_argument(
        "--skip-reviews",
        action="store_true",
        help=(
            "load movies only; skip the review loader entirely (leaves the "
            "reviews collection untouched). Default off preserves movies+reviews."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    settings = Settings()
    overrides: dict = {}
    if args.embedder is not None:
        overrides["embedder"] = args.embedder
    if args.chunk_max_tokens is not None:
        overrides["chunk_max_tokens"] = args.chunk_max_tokens
    if args.chunk_overlap_tokens is not None:
        overrides["chunk_overlap_tokens"] = args.chunk_overlap_tokens
    if args.embed_text_recipe is not None:
        overrides["embed_text_recipe"] = args.embed_text_recipe
    if overrides:
        settings = settings.model_copy(update=overrides)

    explicit_tmdb_ids: list[int] | None = None
    corpus_mode: bool = getattr(args, "corpus", False)

    if corpus_mode:
        # Corpus mode: ingest an explicit id list into the production (default)
        # collections.  sample stays False — no calib_ namespace.
        if args.tmdb_ids:
            explicit_tmdb_ids = [int(x) for x in args.tmdb_ids.split(",") if x.strip()]
        else:
            from ingestion.scripts.build_corpus_sample import load_corpus

            explicit_tmdb_ids = load_corpus()
        # sample must be False so the pipeline targets unsuffixed collections.
        settings = settings.model_copy(update={"sample": False})
    elif args.tmdb_ids:
        explicit_tmdb_ids = [int(x) for x in args.tmdb_ids.split(",") if x.strip()]
        # Any non-corpus explicit id list lives in the disposable calib_ namespace.
        settings = settings.model_copy(update={"sample": True})
    elif args.golden_sample:
        from ingestion.scripts.build_calibration_sample import build_sample

        explicit_tmdb_ids = build_sample(distractors=300)
        # Golden sample also lives in the calib_ namespace.
        settings = settings.model_copy(update={"sample": True})

    # In corpus mode targeting the plain collections is the point — don't warn.
    if settings._is_default_variant() and not corpus_mode:
        _logger.warning(
            '{"step":"warn","msg":"all params are defaults — targeting plain collections; '
            'use --embedder or --chunk-* to target a variant"}'
        )

    _logger.info(
        '{"step":"experiment_start","embedder":"%s","movies_collection":"%s"}',
        settings.embedder,
        settings.movies_collection,
    )

    if args.drop_variant:
        client = get_qdrant_client(os.environ["QDRANT_URL"], os.environ["QDRANT_API_KEY"])
        drop_variant(client, settings)
        _logger.info('{"step":"drop_variant_done"}')
        return

    run_pipeline(
        tmdb_api_key=os.environ["TMDB_API_KEY"],
        openai_api_key=os.environ["OPENAI_API_KEY"],
        qdrant_url=os.environ["QDRANT_URL"],
        qdrant_api_key=os.environ["QDRANT_API_KEY"],
        settings=settings,
        rebuild=args.rebuild,
        refresh_taste=args.refresh_taste,
        skip_taste=args.skip_taste,
        explicit_tmdb_ids=explicit_tmdb_ids,
        resume=args.resume,
        workers=args.workers,
        skip_reviews=args.skip_reviews,
    )


if __name__ == "__main__":
    main()
