from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from uuid import UUID

import pytest
from qdrant_client import models

from consumer_finance_rag.config import MLflowSettings, RetrievalSettings
from consumer_finance_rag.context import ContextBuilder
from consumer_finance_rag.evaluation.managed_rag import (
    CollectingRetrievalTelemetrySink,
    ManagedEvaluationDatasetError,
    ManagedRAGEvaluationDataset,
    evaluate_managed_rag,
    load_managed_rag_dataset,
    validate_managed_dataset_compatibility,
    write_managed_rag_report,
)
from consumer_finance_rag.evaluation.mlflow_genai import MLflowJudgeResults
from consumer_finance_rag.evaluation.models import (
    EvaluationCase,
    JudgeMetricValue,
    RAGCaseResult,
)
from consumer_finance_rag.lineage.managed_retrieval import ManagedRetrievalRuntime
from consumer_finance_rag.models import Chunk, RetrievedChunk
from consumer_finance_rag.observability import RetrievalTelemetryContext
from consumer_finance_rag.pipeline import RAGPipeline
from consumer_finance_rag.providers import LLMResult
from consumer_finance_rag.retrieval import HybridRetrievalService
from consumer_finance_rag.sparse import BM25SparseEncoder

DATASET_PATH = Path("data/evaluation/managed_public_cases.json")
NOW = datetime(2026, 9, 23, 1, 2, 3, tzinfo=UTC)


class FakeEmbedding:
    model_name = "test/embedding"

    def __init__(self, questions: Sequence[str]) -> None:
        self._positions = {question: position for position, question in enumerate(questions, 1)}
        self.query_calls: list[str] = []

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [[float(position)] for position, _text in enumerate(texts, 1)]

    def embed_query(self, text: str) -> list[float]:
        self.query_calls.append(text)
        return [float(self._positions[text])]


class FakeReranker:
    model_name = "test/reranker"

    def __init__(self) -> None:
        self.text_calls: list[tuple[str, ...]] = []

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        del query
        self.text_calls.append(tuple(item.chunk.text for item in candidates))
        return [
            item.model_copy(update={"rerank_score": 1.0})
            for item in candidates[:top_k]
        ]


class ReadOnlyStore:
    def __init__(
        self,
        cases: Sequence[Any],
        candidates: Mapping[str, RetrievedChunk],
    ) -> None:
        self._queries = {float(position): case.question for position, case in enumerate(cases, 1)}
        self._candidates = dict(candidates)
        self.filters: list[models.Filter | None] = []
        self.write_count = 0

    def indexed_chunk_hashes(self) -> dict[str, str]:
        return {
            item.chunk.chunk_id: item.chunk.chunk_hash
            for item in self._candidates.values()
        }

    def search_dense(
        self,
        vector: list[float],
        limit: int,
        *,
        query_filter: models.Filter | None = None,
    ) -> list[RetrievedChunk]:
        self.filters.append(query_filter)
        question = self._queries[vector[0]]
        return [self._candidates[question].model_copy(update={"dense_rank": 1})][:limit]

    def search_sparse(
        self,
        query: str,
        encoder: BM25SparseEncoder,
        limit: int,
        *,
        query_filter: models.Filter | None = None,
    ) -> list[RetrievedChunk]:
        del encoder
        self.filters.append(query_filter)
        return [self._candidates[query].model_copy(update={"sparse_rank": 1})][:limit]


class FakeLLM:
    model_name = "test/bedrock-generation"

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(self, system_prompt: str, user_prompt: str) -> LLMResult:
        del system_prompt
        self.prompts.append(user_prompt)
        assert "alice@example.com" not in user_prompt
        return LLMResult(
            text="Reviewed public answer [C1].",
            model_id=self.model_name,
            response_metadata={
                "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15}
            },
        )


class FakeRagasEvaluator:
    enabled = True

    def evaluate(
        self,
        results: Sequence[RAGCaseResult],
    ) -> dict[str, dict[str, JudgeMetricValue]]:
        return {
            result.case_id: {
                "faithfulness": JudgeMetricValue(
                    value=0.75,
                    reason="must not be serialized: alice@example.com",
                )
            }
            for result in results
        }


def _candidate(case: Any, *, pii: bool) -> RetrievedChunk:
    text = (
        "Reviewed public evidence with contact alice@example.com"
        if pii
        else "Reviewed public evidence about the Closing Disclosure."
    )
    return RetrievedChunk(
        chunk=Chunk(
            chunk_id=case.expected_chunk_ids[0],
            document_id=case.expected_document_ids[0],
            document_version="fixture-reviewed-2026-09-08",
            text=text,
            ordinal=0,
            chunk_hash="a" * 64,
            metadata={
                "source_id": case.expected_source_ids[0],
                "title": "Reviewed PUBLIC source",
                "source_url": "https://example.test/public",
                "category": case.category,
                "access_visibility": "PUBLIC",
                "allowed_groups": [],
                "contains_pii": pii,
                "pii_types": ["EMAIL"] if pii else [],
                "pii_detector_version": "deterministic-pii-v1",
            },
        ),
        score=1.0,
    )


def _runtime() -> tuple[
    ManagedRAGEvaluationDataset,
    ManagedRetrievalRuntime,
    CollectingRetrievalTelemetrySink,
    ReadOnlyStore,
    FakeLLM,
]:
    dataset = load_managed_rag_dataset(DATASET_PATH)
    candidates = {
        case.question: _candidate(case, pii=position == 0)
        for position, case in enumerate(dataset.cases)
    }
    store = ReadOnlyStore(dataset.cases, candidates)
    embedding = FakeEmbedding([case.question for case in dataset.cases])
    reranker = FakeReranker()
    telemetry = CollectingRetrievalTelemetrySink()
    service = HybridRetrievalService(
        embedding_provider=embedding,
        reranker=reranker,
        store=store,
        sparse_encoder=BM25SparseEncoder.fit(
            [candidate.chunk.text for candidate in candidates.values()]
        ),
        settings=RetrievalSettings(
            dense_candidates=2,
            sparse_candidates=2,
            fused_candidates=2,
            final_contexts=2,
        ),
        collection_name="consumer_finance_managed_g000007",
        access_control_enabled=True,
        pii_handling_enabled=True,
        telemetry_context=RetrievalTelemetryContext(
            profile="managed",
            index_state_id=UUID(int=7),
            generation=7,
            qdrant_collection="consumer_finance_managed_g000007",
        ),
        telemetry_sink=telemetry,
    )
    managed = ManagedRetrievalRuntime(
        index_state_id=UUID(int=7),
        generation=7,
        qdrant_collection="consumer_finance_managed_g000007",
        snapshot_hash="snapshot-7",
        embedding_fingerprint="embedding-7",
        chunking_fingerprint="chunking-7",
        index_configuration_fingerprint="configuration-7",
        document_count=2,
        chunk_count=2,
        retrieval_service=service,
    )
    llm = FakeLLM()
    return dataset, managed, telemetry, store, llm


def _pipeline(runtime: ManagedRetrievalRuntime, llm: FakeLLM) -> RAGPipeline:
    service = runtime.retrieval_service
    return RAGPipeline(
        retrieval=service,
        context_builder=ContextBuilder(4000),
        llm=llm,
        embedding_provider=service.embedding_provider,
        reranker=service.reranker,
    )


def test_committed_dataset_is_reviewed_public_and_versioned() -> None:
    dataset = load_managed_rag_dataset(DATASET_PATH)

    assert dataset.format_version == "managed-rag-evaluation-dataset-v1"
    assert dataset.status == "reviewed"
    assert dataset.privacy_classification == "PUBLIC_SYNTHETIC_REVIEWED"
    assert len(dataset.cases) >= 2
    assert all(case.review_status == "reviewed" for case in dataset.cases)
    assert all(case.authorization.visibility == "PUBLIC" for case in dataset.cases)
    assert all(not case.authorization.groups for case in dataset.cases)


def test_unreviewed_or_restricted_dataset_is_rejected(tmp_path: Path) -> None:
    payload = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    payload["cases"][0]["review_status"] = "review_required"
    path = tmp_path / "unreviewed.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ManagedEvaluationDatasetError):
        load_managed_rag_dataset(path)

    payload = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    payload["cases"][0]["authorization"] = {
        "visibility": "PUBLIC",
        "principal_id": None,
        "groups": ["restricted-group"],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ManagedEvaluationDatasetError):
        load_managed_rag_dataset(path)


def test_incompatible_active_generation_is_rejected_before_retrieval() -> None:
    dataset, runtime, _telemetry, store, _llm = _runtime()
    store._candidates.pop(dataset.cases[0].question)

    with pytest.raises(ManagedEvaluationDatasetError, match="absent"):
        validate_managed_dataset_compatibility(
            dataset,
            runtime,
            configured_corpus_key=dataset.corpus_key,
        )
    embedding = cast(FakeEmbedding, runtime.retrieval_service.embedding_provider)
    assert embedding.query_calls == []
    assert store.write_count == 0


def test_retrieval_only_evaluation_is_pinned_public_redacted_and_read_only() -> None:
    dataset, runtime, telemetry, store, _llm = _runtime()

    report = evaluate_managed_rag(
        runtime,
        dataset,
        configured_corpus_key=dataset.corpus_key,
        telemetry_sink=telemetry,
        generated_at=NOW,
    )

    assert report.mode == "retrieval_only"
    assert report.generation == 7
    assert report.index_state_id == UUID(int=7)
    assert report.qdrant_collection.endswith("g000007")
    assert report.retrieval_quality.recall_at_k == 1.0
    assert report.retrieval_latency.request_count == 2
    assert report.retrieval_latency.mean_total_ms is not None
    assert report.answer_quality.generation.status == "NOT_REQUESTED"
    assert report.answer_quality.mlflow_genai.status == "NOT_REQUESTED"
    assert report.answer_quality.ragas.status == "NOT_REQUESTED"
    assert report.answer_cases == ()
    assert all(query_filter is not None for query_filter in store.filters)
    assert store.write_count == 0


def test_generated_answers_use_pipeline_acl_and_redacted_context(tmp_path: Path) -> None:
    dataset, runtime, telemetry, store, llm = _runtime()

    report = evaluate_managed_rag(
        runtime,
        dataset,
        configured_corpus_key=dataset.corpus_key,
        pipeline=_pipeline(runtime, llm),
        generation_model=llm.model_name,
        telemetry_sink=telemetry,
        generated_at=NOW,
    )
    path = write_managed_rag_report(report, tmp_path / "managed.json")
    serialized = path.read_text(encoding="utf-8")

    assert report.mode == "generated_answers"
    assert report.answer_quality.generation.status == "COMPLETED"
    assert report.answer_quality.generation.generated_case_count == 2
    assert report.answer_quality.generation.token_usage.total_tokens == 30
    assert len(report.answer_cases) == 2
    assert any("[REDACTED:EMAIL]" in prompt for prompt in llm.prompts)
    assert all(query_filter is not None for query_filter in store.filters)
    assert "alice@example.com" not in serialized
    assert "Reviewed public answer" not in serialized
    assert dataset.cases[0].question not in serialized
    assert "restricted-group" not in serialized
    assert store.write_count == 0


def test_explicit_judges_reuse_batch_adapters_and_keep_rationales_out_of_report(
    tmp_path: Path,
) -> None:
    dataset, runtime, telemetry, _store, llm = _runtime()
    calls = {"mlflow": 0, "ragas": 0}

    def mlflow_runner(
        cases: Sequence[EvaluationCase],
        results: Sequence[RAGCaseResult],
        settings: MLflowSettings,
    ) -> MLflowJudgeResults:
        del cases, settings
        calls["mlflow"] += 1
        return MLflowJudgeResults(
            aggregate={"correctness": 0.8},
            by_case={
                result.case_id: {
                    "correctness": JudgeMetricValue(
                        value=0.8,
                        reason="must not persist alice@example.com",
                    )
                }
                for result in results
            },
            run_id="judge-run-1",
        )

    def ragas_builder(**kwargs: Any) -> FakeRagasEvaluator:
        assert kwargs["enabled"] is True
        assert kwargs["judge_model"].startswith("bedrock:/")
        calls["ragas"] += 1
        return FakeRagasEvaluator()

    report = evaluate_managed_rag(
        runtime,
        dataset,
        configured_corpus_key=dataset.corpus_key,
        pipeline=_pipeline(runtime, llm),
        generation_model=llm.model_name,
        judge_model="bedrock:/test-judge",
        with_mlflow_judges=True,
        with_ragas=True,
        mlflow_settings=MLflowSettings(judge_model="bedrock:/test-judge"),
        telemetry_sink=telemetry,
        trace_id_provider=lambda: "trace-1",
        mlflow_judge_runner=mlflow_runner,
        ragas_builder=ragas_builder,
        generated_at=NOW,
    )
    serialized = write_managed_rag_report(
        report, tmp_path / "judged.json"
    ).read_text(encoding="utf-8")

    assert calls == {"mlflow": 1, "ragas": 1}
    assert report.answer_quality.mlflow_genai.status == "COMPLETED"
    assert report.answer_quality.mlflow_genai.run_id == "judge-run-1"
    assert report.answer_quality.ragas.status == "COMPLETED"
    assert all(case.scoring.mlflow_genai == "COMPLETED" for case in report.answer_cases)
    assert all(case.scoring.ragas == "COMPLETED" for case in report.answer_cases)
    assert "must not persist" not in serialized
    assert "alice@example.com" not in serialized


def test_requested_judge_failure_is_not_reported_as_pass_or_leaked() -> None:
    dataset, runtime, telemetry, _store, llm = _runtime()

    def failing_runner(*_args: object, **_kwargs: object) -> MLflowJudgeResults:
        raise RuntimeError("restricted context secret@example.com")

    report = evaluate_managed_rag(
        runtime,
        dataset,
        configured_corpus_key=dataset.corpus_key,
        pipeline=_pipeline(runtime, llm),
        generation_model=llm.model_name,
        judge_model="bedrock:/test-judge",
        with_mlflow_judges=True,
        mlflow_settings=MLflowSettings(judge_model="bedrock:/test-judge"),
        telemetry_sink=telemetry,
        mlflow_judge_runner=failing_runner,
        generated_at=NOW,
    )
    serialized = report.model_dump_json()

    assert report.answer_quality.mlflow_genai.status == "FAILED"
    assert "RuntimeError" in (report.answer_quality.mlflow_genai.reason or "")
    assert all(case.scoring.mlflow_genai == "FAILED" for case in report.answer_cases)
    assert "secret@example.com" not in serialized


def test_mlflow_groundedness_is_explicitly_unavailable_without_all_traces() -> None:
    dataset, runtime, telemetry, _store, llm = _runtime()

    def runner(
        cases: Sequence[EvaluationCase],
        results: Sequence[RAGCaseResult],
        settings: MLflowSettings,
    ) -> MLflowJudgeResults:
        del cases, settings
        return MLflowJudgeResults(
            aggregate={"correctness": 1.0},
            by_case={result.case_id: {} for result in results},
            run_id="run-no-traces",
        )

    report = evaluate_managed_rag(
        runtime,
        dataset,
        configured_corpus_key=dataset.corpus_key,
        pipeline=_pipeline(runtime, llm),
        generation_model=llm.model_name,
        judge_model="bedrock:/test-judge",
        with_mlflow_judges=True,
        mlflow_settings=MLflowSettings(judge_model="bedrock:/test-judge"),
        telemetry_sink=telemetry,
        mlflow_judge_runner=runner,
        generated_at=NOW,
    )

    assert report.answer_quality.mlflow_genai.status == "COMPLETED"
    assert "retrieval_groundedness" in report.answer_quality.mlflow_genai.skipped_metrics
    assert all(case.scoring.mlflow_genai == "UNAVAILABLE" for case in report.answer_cases)
