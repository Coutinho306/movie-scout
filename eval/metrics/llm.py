"""LLM-as-judge metrics: RAGAS wrappers + TasteMatch."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from langchain_openai import OpenAIEmbeddings
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

TASTE_PROFILE_PATH = Path("data/taste_profile.json")
JUDGE_PROMPT_PATH = Path(__file__).parent.parent / "prompts/judge_taste_match.md"


class _LLMMetricSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    model_judge: str = "gpt-4o-mini"
    embedder: str = "openai-3-small"


def _load_centroid() -> np.ndarray:
    data = json.loads(TASTE_PROFILE_PATH.read_text())
    return np.array(data["centroid"], dtype=np.float32)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def taste_match(answer_text: str) -> float:
    """Cosine similarity between answer embedding and taste profile centroid."""
    _settings = _LLMMetricSettings()
    embedder = OpenAIEmbeddings(model="text-embedding-3-small")
    answer_vec = np.array(embedder.embed_query(answer_text), dtype=np.float32)
    centroid = _load_centroid()
    return _cosine(answer_vec, centroid)


def ragas_faithfulness(question: str, answer: str, contexts: list[str]) -> float:
    """Run RAGAS Faithfulness on one QA pair. Returns score 0-1."""
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import faithfulness

        ds = Dataset.from_dict(
            {
                "question": [question],
                "answer": [answer],
                "contexts": [contexts],
            }
        )
        result = evaluate(ds, metrics=[faithfulness])
        return float(result["faithfulness"])
    except Exception as exc:  # noqa: BLE001 — tolerate RAGAS API drift
        logger.warning("RAGAS faithfulness failed: %s", exc)
        return float("nan")


def ragas_answer_relevancy(question: str, answer: str, contexts: list[str]) -> float:
    """Run RAGAS AnswerRelevancy on one QA pair. Returns score 0-1."""
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy

        ds = Dataset.from_dict(
            {
                "question": [question],
                "answer": [answer],
                "contexts": [contexts],
            }
        )
        result = evaluate(ds, metrics=[answer_relevancy])
        return float(result["answer_relevancy"])
    except Exception as exc:  # noqa: BLE001 — tolerate RAGAS API drift
        logger.warning("RAGAS answer_relevancy failed: %s", exc)
        return float("nan")


def hallucination_rate(tmdb_ids_in_answer: list[int], retrieved_ids: list[int]) -> float:
    """Fraction of cited tmdb_ids not present in retrieved context."""
    if not tmdb_ids_in_answer:
        return 0.0
    retrieved_set = set(retrieved_ids)
    hallucinated = sum(1 for i in tmdb_ids_in_answer if i not in retrieved_set)
    return hallucinated / len(tmdb_ids_in_answer)
