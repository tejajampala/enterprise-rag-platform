from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from uuid import UUID

import pytest
from pydantic import ValidationError

from consumer_finance_rag.access import AccessContext
from consumer_finance_rag.config import RetrievalSettings
from consumer_finance_rag.evaluation.models import (
    EvaluationCase,
    SeedEvaluationSet,
)
from consumer_finance_rag.lineage.managed_retrieval import ManagedRetrievalRuntime
from consumer_finance_rag.models import Chunk, RetrievalMetadata, RetrievedChunk
from consumer_finance_rag.retrieval import HybridRetrievalService, RetrievalTrace
from consumer_finance_rag.retrieval_quality import (
    REPORT_FORMAT_VERSION,
    RetrievalEvaluationCase,
    RetrievalQualityThresholds,
    compare_quality_reports,
    evaluate_managed_retrieval,
    evaluate_quality_thresholds,
    evaluation_cases_from_seed,
    score_retrieval_case,
    write_retrieval_quality_report,
)

NOW = datetime(2026, 9, 22, 1, 2, 3, tzinfo=UTC)


class FakeReranker:
    model_name = "test/reranker"


class ScriptedRetrievalService:
    def __init__(
        self,
        outputs: Mapping[tuple[str, frozenset[str]], Sequence[RetrievedChunk]],
        *,
        on_retrieve: object | None = None,
    ) -> None:
        self.settings = RetrievalSettings(
            dense_candidates=8,
            sparse_candidates=7,
            fused_candidates=6,
            final_contexts=5,
            rrf_k=42,
            hnsw_ef=64,
        )
        self.reranker = FakeReranker()
        self.access_control_enabled = True
        self.outputs = {key: list(value) for key, value in outputs.items()}
        self.access_contexts: list[AccessContext] = []
        self.retrieval_count = 0
        self.on_retrieve = on_retrieve

    def retrieve_debug(
        self,
        query: str,
        top_k: int | None = None,
        *,
        access_context: AccessContext | None = None,
    ) -> RetrievalTrace:
        context = access_context or AccessContext()
        self.access_contexts.append(context)
        self.retrieval_count += 1
        if callable(self.on_retrieve):
            self.on_retrieve(self.retrieval_count)
        selected = self.outputs.get((query, context.groups), [])
        reranked = list(selected[:top_k])
        metadata = RetrievalMetadata(
            dense_candidates=len(reranked),
            sparse_candidates=len(reranked),
            fused_candidates=len(reranked),
            reranked=len(reranked),
            final_contexts=len(reranked),
        )
        return RetrievalTrace(
            dense=reranked,
            sparse=reranked,
            fused=reranked,
            reranked=reranked,
            metadata=metadata,
        )


def retrieved(
    chunk_id: str,
    source_id: str,
    *,
    document_id: str | None = None,
    text: str = "raw context that must never enter a report",
    metadata: Mapping[str, object] | None = None,
) -> RetrievedChunk:
    payload: dict[str, object] = {
        "source_id": source_id,
        "credential": "do-not-serialize-this-secret",
    }
    payload.update(metadata or {})
    return RetrievedChunk(
        chunk=Chunk(
            chunk_id=chunk_id,
            document_id=document_id or f"document-{source_id}",
            document_version="v1",
            text=text,
            ordinal=0,
            chunk_hash="a" * 64,
            metadata=payload,
        ),
        score=1.0,
    )


def runtime(
    generation: int,
    outputs: Mapping[tuple[str, frozenset[str]], Sequence[RetrievedChunk]],
    *,
    on_retrieve: object | None = None,
) -> tuple[ManagedRetrievalRuntime, ScriptedRetrievalService]:
    service = ScriptedRetrievalService(outputs, on_retrieve=on_retrieve)
    managed = ManagedRetrievalRuntime(
        index_state_id=UUID(int=generation),
        generation=generation,
        qdrant_collection=f"consumer_finance_managed_g{generation:06d}",
        snapshot_hash=f"snapshot-{generation}",
        embedding_fingerprint=f"embedding-{generation}",
        chunking_fingerprint=f"chunking-{generation}",
        index_configuration_fingerprint=f"index-{generation}",
        document_count=2,
        chunk_count=3,
        retrieval_service=cast(HybridRetrievalService, service),
    )
    return managed, service


def source_case(
    case_id: str,
    query: str,
    source_id: str,
    *,
    groups: frozenset[str] = frozenset(),
    expected_access: bool = True,
) -> RetrievalEvaluationCase:
    return RetrievalEvaluationCase(
        case_id=case_id,
        query=query,
        expected_source_ids=(source_id,),
        access_context=AccessContext(groups=groups),
        expected_access=expected_access,
    )


def test_perfect_and_missed_retrieval_metrics_and_managed_identity() -> None:
    managed, _service = runtime(
        4,
        {
            ("found", frozenset()): [retrieved("chunk-target", "target")],
            ("missed", frozenset()): [retrieved("chunk-other", "other")],
        },
    )
    report = evaluate_managed_retrieval(
        managed,
        [source_case("found", "found", "target")],
        k=5,
        generated_at=NOW,
    )

    assert report.generation == 4
    assert report.index_state_id == UUID(int=4)
    assert report.qdrant_collection.endswith("g000004")
    assert report.recall_at_k == report.mrr == report.hit_rate == 1.0
    assert report.missed_case_count == 0
    assert report.cases[0].expected_rank == 1
    assert report.cases[0].final_context_count == 1

    missed = evaluate_managed_retrieval(
        managed,
        [source_case("missed", "missed", "target")],
        k=5,
        generated_at=NOW,
    )
    assert missed.recall_at_k == missed.mrr == missed.hit_rate == 0.0
    assert missed.missed_case_count == 1
    assert missed.cases[0].expected_rank is None


def test_reciprocal_rank_and_duplicates_reuse_phase2_metric_semantics() -> None:
    case = source_case("rank", "rank", "target")
    result = score_retrieval_case(
        case,
        [
            retrieved("other-1", "other"),
            retrieved("other-2", "other"),
            retrieved("target", "target"),
        ],
        generation=1,
        qdrant_collection="collection",
        k=3,
    )

    assert result.expected_rank == 2
    assert result.reciprocal_rank == 0.5
    assert result.recall_at_k == 1.0


def test_mrr_aggregates_case_reciprocal_ranks() -> None:
    managed, _service = runtime(
        1,
        {
            ("first", frozenset()): [retrieved("target-a", "a")],
            ("second", frozenset()): [
                retrieved("other", "other"),
                retrieved("target-b", "b"),
            ],
        },
    )
    report = evaluate_managed_retrieval(
        managed,
        [source_case("a", "first", "a"), source_case("b", "second", "b")],
        generated_at=NOW,
    )

    assert report.mrr == 0.75


def test_scoring_can_use_stable_source_document_or_chunk_identity() -> None:
    item = retrieved("new-chunk", "stable-source", document_id="stable-document")
    cases = [
        RetrievalEvaluationCase(
            case_id="source",
            query="q",
            expected_source_ids=("stable-source",),
        ),
        RetrievalEvaluationCase(
            case_id="document",
            query="q",
            expected_document_ids=("stable-document",),
        ),
        RetrievalEvaluationCase(
            case_id="chunk",
            query="q",
            expected_chunk_ids=("new-chunk",),
        ),
    ]

    for case in cases:
        result = score_retrieval_case(
            case,
            [item],
            generation=1,
            qdrant_collection="collection",
            k=1,
        )
        assert result.hit_at_k


def test_cases_require_one_unambiguous_identity_family() -> None:
    with pytest.raises(ValidationError, match="exactly one"):
        RetrievalEvaluationCase(case_id="none", query="query")
    with pytest.raises(ValidationError, match="exactly one"):
        RetrievalEvaluationCase(
            case_id="two",
            query="query",
            expected_source_ids=("source",),
            expected_chunk_ids=("chunk",),
        )


def test_acl_expected_access_semantics_and_group_forwarding() -> None:
    fraud = frozenset({"fraud-analysts"})
    wrong = frozenset({"customer-service"})
    protected = retrieved("fraud-chunk", "fraud-source")
    managed, service = runtime(
        2,
        {
            ("public", frozenset()): [],
            ("allowed", fraud): [protected],
            ("wrong", wrong): [],
        },
    )
    cases = [
        source_case(
            "public-inaccessible",
            "public",
            "fraud-source",
            expected_access=False,
        ),
        source_case("allowed", "allowed", "fraud-source", groups=fraud),
        source_case(
            "wrong-group",
            "wrong",
            "fraud-source",
            groups=wrong,
            expected_access=False,
        ),
    ]
    report = evaluate_managed_retrieval(managed, cases, generated_at=NOW)

    assert report.hit_rate == 1.0
    assert all(case.access_expectation_met for case in report.cases)
    assert report.cases[0].expected_rank is None
    assert [context.groups for context in service.access_contexts] == [
        frozenset(),
        fraud,
        wrong,
    ]


def test_acl_leak_is_scored_as_a_failure() -> None:
    result = score_retrieval_case(
        source_case(
            "leak",
            "query",
            "protected",
            expected_access=False,
        ),
        [retrieved("protected-chunk", "protected")],
        generation=1,
        qdrant_collection="collection",
        k=5,
    )

    assert not result.hit_at_k
    assert not result.access_expectation_met
    assert result.expected_rank == 1


def test_pii_redaction_does_not_affect_identity_scoring_or_report_privacy(
    tmp_path: Path,
) -> None:
    redacted = retrieved(
        "pii-chunk",
        "pii-source",
        text="Contact [EMAIL] or [PHONE] using card [CARD_NUMBER]",
        metadata={"principal_secret": "never serialize"},
    )
    secret_query = "email alice@example.test about 4111 1111 1111 1111"
    group = frozenset({"fraud-analysts"})
    managed, _service = runtime(3, {(secret_query, group): [redacted]})
    report = evaluate_managed_retrieval(
        managed,
        [source_case("pii", secret_query, "pii-source", groups=group)],
        generated_at=NOW,
    )
    output = write_retrieval_quality_report(
        report,
        tmp_path / "quality.json",
    )
    serialized = output.read_text(encoding="utf-8")

    assert report.hit_rate == 1.0
    assert secret_query not in serialized
    assert "raw context" not in serialized
    assert "[EMAIL]" not in serialized
    assert "fraud-analysts" not in serialized
    assert "principal_secret" not in serialized
    assert "do-not-serialize-this-secret" not in serialized


def test_report_format_is_deterministic_except_timestamp() -> None:
    managed, _service = runtime(
        1,
        {("query", frozenset()): [retrieved("target", "source")]},
    )
    case = source_case("case", "query", "source")
    first = evaluate_managed_retrieval(managed, [case], generated_at=NOW)
    second = evaluate_managed_retrieval(
        managed,
        [case],
        generated_at=datetime(2026, 9, 23, tzinfo=UTC),
    )
    first_json = json.loads(first.model_dump_json())
    second_json = json.loads(second.model_dump_json())
    first_json.pop("generated_at")
    second_json.pop("generated_at")

    assert first.format_version == REPORT_FORMAT_VERSION
    assert first_json == second_json
    assert first_json["snapshot_hash"] == "snapshot-1"
    assert first_json["retrieval_settings"]["rrf_k"] == 42


def test_runtime_generation_is_copied_once_for_the_whole_run() -> None:
    external_active = {"generation": 4}

    def change_active(call_count: int) -> None:
        if call_count == 1:
            external_active["generation"] = 5

    managed, _service = runtime(
        4,
        {
            ("one", frozenset()): [retrieved("one", "one")],
            ("two", frozenset()): [retrieved("two", "two")],
        },
        on_retrieve=change_active,
    )
    report = evaluate_managed_retrieval(
        managed,
        [source_case("one", "one", "one"), source_case("two", "two", "two")],
        generated_at=NOW,
    )

    assert external_active["generation"] == 5
    assert report.generation == 4
    assert {case.generation for case in report.cases} == {4}


def test_phase2_seed_cases_are_reused_without_serializing_questions() -> None:
    phase2 = EvaluationCase(
        case_id="phase2",
        question="private evaluation question",
        category="payments/fraud",
        difficulty="easy",
        expected_facts=["fact"],
        reference_answer="answer",
        relevant_document_ids=["document"],
        relevant_chunk_ids=["chunk"],
    )
    dataset = SeedEvaluationSet(
        name="seed",
        version="1",
        source_corpus="fixture",
        cases=[phase2],
    )
    converted = evaluation_cases_from_seed(dataset)

    assert converted[0].query == phase2.question
    assert converted[0].expected_chunk_ids == ("chunk",)
    assert "private evaluation question" not in converted[0].model_dump_json()


def test_generation_comparison_classifies_regressions_and_recoveries() -> None:
    cases = [source_case("rank", "rank", "target"), source_case("miss", "miss", "target")]
    baseline_runtime, _ = runtime(
        1,
        {
            ("rank", frozenset()): [retrieved("target-rank", "target")],
            ("miss", frozenset()): [retrieved("target-miss", "target")],
        },
    )
    candidate_runtime, _ = runtime(
        2,
        {
            ("rank", frozenset()): [
                retrieved("other", "other"),
                retrieved("target-rank", "target"),
            ],
            ("miss", frozenset()): [retrieved("other-miss", "other")],
        },
    )
    baseline = evaluate_managed_retrieval(baseline_runtime, cases, generated_at=NOW)
    candidate = evaluate_managed_retrieval(candidate_runtime, cases, generated_at=NOW)
    comparison = compare_quality_reports(baseline, candidate)
    recovery = compare_quality_reports(candidate, baseline)

    assert comparison.baseline_generation == 1
    assert comparison.candidate_generation == 2
    assert comparison.recall_delta == -0.5
    assert comparison.mrr_delta == -0.75
    assert comparison.hit_rate_delta == -0.5
    assert comparison.cases_regressed == ("miss", "rank")
    assert comparison.cases_improved == ()
    assert comparison.newly_missed_cases == ("miss",)
    assert recovery.cases_improved == ("miss", "rank")
    assert recovery.newly_recovered_cases == ("miss",)


def test_threshold_pass_and_fail_have_no_runtime_side_effects() -> None:
    cases = [source_case("case", "query", "target")]
    baseline_runtime, baseline_service = runtime(
        1,
        {("query", frozenset()): [retrieved("target", "target")]},
    )
    candidate_runtime, candidate_service = runtime(
        2,
        {("query", frozenset()): [retrieved("other", "other")]},
    )
    baseline = evaluate_managed_retrieval(baseline_runtime, cases, generated_at=NOW)
    candidate = evaluate_managed_retrieval(candidate_runtime, cases, generated_at=NOW)
    comparison = compare_quality_reports(baseline, candidate)
    calls_before = (baseline_service.retrieval_count, candidate_service.retrieval_count)

    passed = evaluate_quality_thresholds(
        comparison,
        RetrievalQualityThresholds(max_recall_drop=1.0, max_mrr_drop=1.0),
    )
    failed = evaluate_quality_thresholds(
        comparison,
        RetrievalQualityThresholds(max_recall_drop=0.1, max_mrr_drop=0.1),
    )

    assert passed.status == "PASS"
    assert passed.violations == ()
    assert failed.status == "FAIL"
    assert len(failed.violations) == 2
    assert calls_before == (
        baseline_service.retrieval_count,
        candidate_service.retrieval_count,
    )
    assert baseline_runtime.generation == 1
    assert candidate_runtime.generation == 2
