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
) -> list[GoldenQuery]:
    """Ask an LLM to generate one NL query per watchlist film."""
    template = QUERY_GEN_PROMPT.read_text()
    llm = ChatOpenAI(model=model, temperature=0.7)
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
        queries.append(
            GoldenQuery(
                text=query_text,
                target_tmdb_ids={tmdb_id},
                target_titles=[name],
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

    logger.info("Generating NL queries for %d resolved films", len(title_to_id))
    queries = _generate_queries(watchlist, title_to_id, settings.model_orchestrator)

    golden = GoldenSet(
        holdout_tmdb_ids=set(title_to_id.values()),
        queries=queries,
    )
    GOLDEN_CACHE.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN_CACHE.write_text(golden.model_dump_json())
    logger.info("Golden set cached to %s (%d queries)", GOLDEN_CACHE, len(queries))
    return golden
