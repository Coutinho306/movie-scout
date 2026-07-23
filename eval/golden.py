"""Ground truth: build GoldenSet from watchlist.csv + LLM-generated queries."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

GOLDEN_CACHE = Path("data/golden_set.json")
WATCHLIST_CSV = Path("data/letterboxd_export/watchlist.csv")
QUERY_GEN_PROMPT = Path(__file__).parent / "prompts/query_gen.md"

_MOVIES_COLLECTION = "tmdb_movies"


class GoldenQuery(BaseModel):
    text: str
    target_tmdb_ids: set[int]
    target_titles: list[str]


class GoldenSet(BaseModel):
    holdout_tmdb_ids: set[int]
    queries: list[GoldenQuery]


class _EvalSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    tmdb_api_key: str = ""
    model_orchestrator: str = "gpt-4o-mini"


def _scroll_corpus(collection: str = _MOVIES_COLLECTION) -> list[dict]:
    """Scroll all movies from Qdrant, fetching only fields needed for clustering."""
    from retrieval.client import get_qdrant_client

    client = get_qdrant_client()
    movies: list[dict] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=1000,
            offset=offset,
            with_payload=["tmdb_id", "genres", "keywords", "popularity"],
            with_vectors=False,
        )
        movies.extend(p.payload for p in points)
        if offset is None:
            break
    logger.info('{"step":"corpus_scroll","count":%d}', len(movies))
    return movies


def build_relevant_cluster(
    seed_payload: dict,
    corpus: list[dict],
    *,
    tau: float = 0.2,
    n: int = 6,
) -> set[int]:
    """Return the set of tmdb_ids that form the relevance cluster for a seed film.

    Membership criteria:
    - genre overlap >= 1 (hard prefilter)
    - keyword Jaccard >= tau

    The seed is always included. Candidates are ranked by keyword overlap
    descending, then popularity descending, then tmdb_id ascending (deterministic,
    no RNG). The result is capped at n members; if the seed would otherwise be
    excluded by the cap, it is force-included and the last non-seed member is
    dropped.
    """
    seed_id: int = seed_payload["tmdb_id"]
    seed_genres: set[str] = set(seed_payload.get("genres") or [])
    seed_keywords: set[str] = set(seed_payload.get("keywords") or [])

    candidates: list[tuple[float, float, int, int]] = []  # (-overlap, -pop, tmdb_id, idx)

    for idx, movie in enumerate(corpus):
        mid: int = movie.get("tmdb_id")
        if mid is None or mid == seed_id:
            continue

        genres: set[str] = set(movie.get("genres") or [])
        # Hard prefilter: must share at least one genre
        if not (seed_genres & genres):
            continue

        keywords: set[str] = set(movie.get("keywords") or [])
        union = seed_keywords | keywords
        if not union:
            jaccard = 0.0
        else:
            jaccard = len(seed_keywords & keywords) / len(union)

        if jaccard < tau:
            continue

        pop: float = float(movie.get("popularity") or 0.0)
        # Sort key: keyword overlap desc, popularity desc, tmdb_id asc
        overlap = len(seed_keywords & keywords)
        candidates.append((-overlap, -pop, mid, idx))

    candidates.sort()

    cluster: set[int] = {seed_id}
    for _, _, mid, _ in candidates:
        if len(cluster) >= n:
            break
        cluster.add(mid)

    # Seed always present — if cap was hit and seed wasn't in the top N, force it
    # (seed was excluded from the candidate loop, so it's always added above as
    # the initial singleton before the cap logic; this is a no-op guard)
    cluster.add(seed_id)

    return cluster


def _resolve_tmdb_ids(watchlist: pd.DataFrame, api_key: str) -> dict[str, int]:
    """Return {title: tmdb_id} for watchlist films."""
    from agent.tools.tmdb_search import search_tmdb

    result: dict[str, int] = {}
    for _, row in watchlist.iterrows():
        name = str(row["Name"])
        year = int(row["Year"]) if pd.notna(row.get("Year")) else None
        tmdb_id = search_tmdb(name, year)
        if tmdb_id is not None:
            result[name] = tmdb_id
        else:
            logger.warning("Could not resolve TMDB id for: %s (%s)", name, year)
    return result


def _generate_queries(
    watchlist: pd.DataFrame,
    title_to_id: dict[str, int],
    model: str,
    corpus: list[dict] | None = None,
) -> list[GoldenQuery]:
    """Ask an LLM to generate one NL query per watchlist film; build multi-relevant clusters."""
    template = QUERY_GEN_PROMPT.read_text()
    llm = ChatOpenAI(model=model, temperature=0.7)

    # Build lookup: tmdb_id -> payload for cluster construction
    corpus_by_id: dict[int, dict] = {}
    if corpus:
        for movie in corpus:
            mid = movie.get("tmdb_id")
            if mid is not None:
                corpus_by_id[mid] = movie

    queries: list[GoldenQuery] = []
    for _, row in watchlist.iterrows():
        name = str(row["Name"])
        year = int(row["Year"]) if pd.notna(row.get("Year")) else 0
        tmdb_id = title_to_id.get(name)
        if tmdb_id is None:
            continue
        prompt = template.format(title=name, year=year)
        response = llm.invoke(prompt)
        query_text = response.content.strip().strip('"')

        # Build relevance cluster if corpus is available
        if corpus and tmdb_id in corpus_by_id:
            cluster = build_relevant_cluster(corpus_by_id[tmdb_id], corpus)
        else:
            cluster = {tmdb_id}

        # target_titles: seed title first, then any additional cluster member titles
        cluster_titles = [name]
        if corpus:
            for mid in sorted(cluster - {tmdb_id}):
                if mid in corpus_by_id:
                    title = corpus_by_id[mid].get("title", str(mid))
                    cluster_titles.append(title)

        queries.append(
            GoldenQuery(
                text=query_text,
                target_tmdb_ids=cluster,
                target_titles=cluster_titles,
            )
        )
    return queries


def build_golden_set(force: bool = False) -> GoldenSet:
    """Build (or load cached) GoldenSet from watchlist.csv."""
    if not force and GOLDEN_CACHE.exists():
        logger.info("Loading golden set from cache: %s", GOLDEN_CACHE)
        data = json.loads(GOLDEN_CACHE.read_text())
        return GoldenSet.model_validate(data)

    settings = _EvalSettings()
    watchlist = pd.read_csv(WATCHLIST_CSV)
    logger.info("Resolving TMDB ids for %d watchlist films", len(watchlist))
    title_to_id = _resolve_tmdb_ids(watchlist, settings.tmdb_api_key)

    logger.info("Scrolling corpus for clustering")
    corpus = _scroll_corpus()

    logger.info("Generating NL queries for %d resolved films", len(title_to_id))
    queries = _generate_queries(watchlist, title_to_id, settings.model_orchestrator, corpus)

    golden = GoldenSet(
        holdout_tmdb_ids=set(title_to_id.values()),
        queries=queries,
    )
    GOLDEN_CACHE.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN_CACHE.write_text(golden.model_dump_json())
    logger.info("Golden set cached to %s (%d queries)", GOLDEN_CACHE, len(queries))
    return golden
