from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

import pytest
from pydantic_settings import DotEnvSettingsSource

from consumer_finance_rag.evaluation.managed_rag import ManagedRAGEvaluationReport
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


@pytest.fixture
def historical_report_without_usage() -> ManagedRAGEvaluationReport:
    """A content-free historical report from before judge usage was recorded."""

    return ManagedRAGEvaluationReport.model_validate(
        {
            "format_version": "managed-rag-evaluation-report-v1",
            "generated_at": "2026-01-01T00:00:00Z",
            "mode": "generated_answers",
            "corpus_key": "synthetic-public-corpus",
            "snapshot_hash": "synthetic-snapshot",
            "corpus_document_count": 1,
            "corpus_chunk_count": 1,
            "dataset_name": "synthetic-historical-report",
            "dataset_version": "1",
            "dataset_format_version": "managed-rag-evaluation-dataset-v1",
            "dataset_privacy_classification": "PUBLIC_SYNTHETIC_REVIEWED",
            "case_count": 1,
            "generation": 1,
            "index_state_id": "00000000-0000-0000-0000-000000000001",
            "qdrant_collection": "synthetic_managed_g000001",
            "embedding_fingerprint": "synthetic-embedding",
            "chunking_fingerprint": "synthetic-chunking",
            "index_configuration_fingerprint": "synthetic-index-config",
            "models": {
                "embedding_model": "synthetic-embedding-model",
                "reranker_model": "synthetic-reranker-model",
                "generation_model": "synthetic-generation-model",
                "generation_region": "us-east-1",
                "judge_model": "bedrock:/meta.llama3-70b-instruct-v1:0",
                "judge_region": "us-east-1",
            },
            "retrieval_quality": {
                "k": 1,
                "recall_at_k": 1.0,
                "mrr": 1.0,
                "hit_rate": 1.0,
                "missed_case_count": 0,
            },
            "retrieval_cases": [
                {
                    "case_id": "synthetic-case-1",
                    "generation": 1,
                    "qdrant_collection": "synthetic_managed_g000001",
                    "identity_kind": "source_id",
                    "expected_ids": ["synthetic-source"],
                    "expected_access": True,
                    "access_expectation_met": True,
                    "retrieved_identities": [
                        {
                            "source_id": "synthetic-source",
                            "document_id": "synthetic-document",
                            "chunk_id": "synthetic-chunk",
                        }
                    ],
                    "hit_at_k": True,
                    "recall_at_k": 1.0,
                    "reciprocal_rank": 1.0,
                    "expected_rank": 1,
                    "final_context_count": 1,
                    "scenario": "synthetic",
                    "category": "regression",
                }
            ],
            "retrieval_latency": {
                "request_count": 1,
                "mean_total_ms": 1.0,
                "mean_dense_ms": 0.1,
                "mean_sparse_ms": 0.1,
                "mean_fusion_ms": 0.1,
                "mean_reranker_ms": 0.1,
                "mean_pii_ms": 0.1,
            },
            "answer_quality": {
                "generation": {
                    "status": "COMPLETED",
                    "generated_case_count": 1,
                    "mean_latency_ms": 1.0,
                    "token_usage": {
                        "available_case_count": 0,
                        "input_tokens": None,
                        "output_tokens": None,
                        "total_tokens": None,
                    },
                },
                "mlflow_genai": {"status": "NOT_REQUESTED"},
                "ragas": {
                    "status": "COMPLETED",
                    "aggregate_metrics": {"faithfulness": 1.0},
                },
            },
            "answer_cases": [
                {
                    "case_id": "synthetic-case-1",
                    "generation_status": "COMPLETED",
                    "answer_character_count": 0,
                    "citation_count": 0,
                    "cited_chunk_ids": [],
                    "latency_ms": 1.0,
                    "token_usage": None,
                    "scoring": {
                        "mlflow_genai": "NOT_REQUESTED",
                        "ragas": "COMPLETED",
                    },
                    "ragas": {"faithfulness": {"value": 1.0}},
                }
            ],
        }
    )
