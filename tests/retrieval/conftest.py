"""Shared fixtures for retrieval integration tests.

Tests run against live Qdrant + real OpenAI embeddings.
Require QDRANT_URL, QDRANT_API_KEY, OPENAI_API_KEY in .env or environment.
"""

import pytest
from dotenv import load_dotenv

from retrieval.config import RetrievalSettings

load_dotenv()


@pytest.fixture(scope="session")
def settings() -> RetrievalSettings:
    return RetrievalSettings()


@pytest.fixture(scope="session")
def hybrid_settings() -> RetrievalSettings:
    return RetrievalSettings(hybrid=True)
