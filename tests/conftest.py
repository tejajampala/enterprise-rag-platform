from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

import pytest
from pydantic_settings import DotEnvSettingsSource

from consumer_finance_rag.models import Chunk, RetrievedChunk

_SETTINGS_ALIASES = {
    "BEDROCK_GENERATION_MODEL",
    "AWS_REGION",
    "MLFLOW_TRACKING_URI",
    "MLFLOW_GENAI_JUDGE_DEFAULT_MODEL",
    "MLFLOW_GENAI_JUDGE_REGION",
}
@pytest.fixture(autouse=True)
def isolate_tests_from_local_dotenv(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[None]:
    """Keep developer deployment choices from changing deterministic test defaults."""

    _clear_settings_environment()
    monkeypatch.setattr(DotEnvSettingsSource, "__call__", lambda _source: {})
    try:
        yield
    finally:
        _clear_settings_environment()


def _clear_settings_environment() -> None:
    for name in tuple(os.environ):
        if name.startswith("RAG__") or name in _SETTINGS_ALIASES:
            os.environ.pop(name, None)


@pytest.fixture
def make_candidate() -> Any:
    def factory(
        chunk_id: str,
        *,
        score: float = 1.0,
        title: str = "Test document",
        source_url: str | None = "https://example.test/source",
    ) -> RetrievedChunk:
        chunk = Chunk(
            chunk_id=chunk_id,
            document_id="doc-1",
            document_version="v1",
            text=f"Evidence for {chunk_id}",
            ordinal=0,
            chunk_hash="a" * 64,
            metadata={
                "source_id": "source-1",
                "title": title,
                "source_url": source_url,
                "category": "testing",
            },
        )
        return RetrievedChunk(chunk=chunk, score=score)

    return factory
