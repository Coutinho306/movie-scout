"""LLM-as-judge metrics: RAGAS wrappers + TasteMatch."""
from __future__ import annotations

import json
import logging
import sys
import types
from pathlib import Path

import numpy as np
from langchain_openai import OpenAIEmbeddings
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

TASTE_PROFILE_PATH = Path("data/taste_profile.json")
JUDGE_PROMPT_PATH = Path(__file__).parent.parent / "prompts/judge_taste_match.md"


def _ensure_ragas_importable() -> None:
    """Stub out langchain_community.chat_models.vertexai so `import ragas` resolves.

    ragas/llms/base.py unconditionally does
    `from langchain_community.chat_models.vertexai import ChatVertexAI` at module
    scope. Our langchain-community (0.4.2, langchain-1.x era) no longer ships that
    submodule, so plain `import ragas` raises ModuleNotFoundError even though we
    never touch Vertex AI (judge is ChatOpenAI throughout). Confirmed upstream and
    still open on ragas `main`: explodinggradients/ragas#2745, #2741. Registering a
    dummy module satisfies the import; the stub class is never instantiated.
    """
    mod_name = "langchain_community.chat_models.vertexai"
    if mod_name in sys.modules:
        return
    stub = types.ModuleType(mod_name)
    stub.ChatVertexAI = type("ChatVertexAI", (), {})  # never instantiated
    sys.modules[mod_name] = stub


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
    _ensure_ragas_importable()
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
    # result["metric_name"] is a per-row list; we always score exactly one row.
    return float(result["faithfulness"][0])


def ragas_answer_relevancy(question: str, answer: str, contexts: list[str]) -> float:
    """Run RAGAS AnswerRelevancy on one QA pair. Returns score 0-1."""
    _ensure_ragas_importable()
    from datasets import Dataset
    from ragas import evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.metrics import answer_relevancy

    ds = Dataset.from_dict(
        {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
        }
    )
    # answer_relevancy needs an embeddings model; ragas' default auto-factory
    # constructs its "modern" OpenAIEmbeddings without a client (a ragas-internal
    # bug — `embed_query` ends up missing). Pass our own langchain embeddings
    # explicitly via the legacy wrapper to bypass the broken auto-factory.
    embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))
    result = evaluate(ds, metrics=[answer_relevancy], embeddings=embeddings)
    # result["metric_name"] is a per-row list; we always score exactly one row.
    return float(result["answer_relevancy"][0])


def hallucination_rate(tmdb_ids_in_answer: list[int], retrieved_ids: list[int]) -> float:
    """Fraction of cited tmdb_ids not present in retrieved context."""
    if not tmdb_ids_in_answer:
        return 0.0
    retrieved_set = set(retrieved_ids)
    hallucinated = sum(1 for i in tmdb_ids_in_answer if i not in retrieved_set)
    return hallucinated / len(tmdb_ids_in_answer)
