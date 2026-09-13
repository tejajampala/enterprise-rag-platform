from __future__ import annotations

from typing import Any

import pytest

from consumer_finance_rag.models import Chunk, RetrievedChunk


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
