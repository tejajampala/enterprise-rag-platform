from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID

from consumer_finance_rag.drift import (
    AssessmentStatus,
    DriftFindingCategory,
    DriftThresholds,
    EvaluationIdentity,
    GenerationIdentity,
    GenerationSnapshot,
    OperationalTelemetrySample,
    OperationalTelemetryWindow,
    PhysicalValidationSummary,
    RetrievalRegressionAssessment,
    SourceDocumentSnapshot,
    assess_answer_regression,
    assess_operational_trends,
    assess_retrieval_regression,
    assess_structural_drift,
    build_drift_report,
    load_drift_report,
    log_drift_report,
    write_drift_report,
)
from consumer_finance_rag.evaluation.managed_rag import ManagedRAGEvaluationReport
from consumer_finance_rag.retrieval_quality import (
    RetrievalCaseResult,
    RetrievalQualityMetrics,
    RetrievalQualityReport,
    RetrievalRuntimeConfiguration,
    RetrievedIdentity,
)

NOW = datetime(2026, 9, 24, tzinfo=UTC)


def _configuration(*, pii: str = "detector-v1") -> dict[str, object]:
    return {
        "bm25": {
            "implementation_version": "client-bm25-v1",
            "tokenizer_version": "tokenizer-v1",
            "k1": 1.2,
            "b": 0.75,
        },
        "pii_detector_version": pii,
        "payload_schema_version": "managed-qdrant-payload-v3",
        "point_id_format_version": "point-v1",
    }


def _document(
    source: str,
    *,
    content: str = "content-a",
    metadata: str = "metadata-a",
    category: str = "cards",
    source_version: str = "fixture-reviewed-2026-09-08",
    policy: str = "policy-public",
) -> SourceDocumentSnapshot:
    return SourceDocumentSnapshot(
        source_key=source,
        content_hash=content,
        metadata_hash=metadata,
        category=category,
        source_version=source_version,
        access_policy_fingerprint=policy,
    )


def _snapshot(
    generation: int,
    documents: tuple[SourceDocumentSnapshot, ...],
    *,
    authoritative: bool = True,
    configuration: dict[str, object] | None = None,
) -> GenerationSnapshot:
    return GenerationSnapshot(
        identity=GenerationIdentity(
            generation=generation,
            index_state_id=UUID(int=generation),
            status="ACTIVE" if generation == 2 else "RETIRED",
            qdrant_collection=f"managed_g{generation:06d}",
            snapshot_hash=f"snapshot-{generation}",
            index_configuration_fingerprint=f"configuration-{generation}",
            embedding_fingerprint="embedding-a",
            chunking_fingerprint="chunking-a",
            bm25_state_hash="bm25-a",
            bm25_corpus_hash="corpus-a",
            document_count=len(documents),
            chunk_count=len(documents),
            ingestion_run_id=UUID(int=100 + generation),
            authoritative_snapshot=authoritative,
        ),
        index_configuration=configuration or _configuration(),
        documents=documents,
        membership_count=len(documents),
    )


def _validation(
    generation: int,
    *,
    outcome: str = "PASS",
    physical: bool = True,
    governance: bool = False,
) -> PhysicalValidationSummary:
    return PhysicalValidationSummary(
        generation=generation,
        status="ACTIVE" if generation == 2 else "RETIRED",
        validation_outcome=outcome,  # type: ignore[arg-type]
        classification="FAIL" if outcome == "FAIL" else "PASS",
        physical_present=physical,
        deep_validation=True,
        issue_count=int(outcome == "FAIL"),
        note_count=0,
        membership_count=1,
        physical_point_count=1 if physical else None,
        governance_failure_detected=governance,
    )


def _thresholds(*, samples: int = 3) -> DriftThresholds:
    return DriftThresholds(
        max_recall_drop=0.0,
        max_mrr_drop=0.0,
        max_answer_metric_drop=0.05,
        minimum_operational_samples=samples,
        max_failure_rate_increase=0.1,
        max_mean_latency_increase_ratio=0.2,
    )


def test_identical_generations_have_no_structural_drift() -> None:
    documents = (_document("a", source_version="2026-09-01"),)
    baseline = _snapshot(1, documents)
    candidate = _snapshot(2, documents)

    result = assess_structural_drift(
        baseline,
        candidate,
        baseline_validation=_validation(1),
        candidate_validation=_validation(2),
        assessed_at=NOW,
    )

    assert result.source_changes.model_dump()["new_count"] == 0
    assert result.source_changes.changed_count == 0
    assert result.source_changes.unchanged_count == 1
    assert result.source_changes.deleted_count == 0
    assert result.source_changes.candidate_freshness.status == "AVAILABLE"
    assert result.index_compatibility.comparison_supported is True


def test_content_metadata_governance_and_authoritative_deletion_are_separate() -> None:
    baseline = _snapshot(
        1,
        (
            _document("content"),
            _document("metadata"),
            _document("deleted"),
        ),
    )
    candidate = _snapshot(
        2,
        (
            _document("content", content="content-b"),
            _document(
                "metadata",
                policy="policy-restricted-withheld",
            ),
            _document("new", category="mortgages"),
        ),
    )

    changes = assess_structural_drift(
        baseline,
        candidate,
        baseline_validation=_validation(1),
        candidate_validation=_validation(2),
        assessed_at=NOW,
    ).source_changes

    assert (changes.new_count, changes.changed_count, changes.deleted_count) == (1, 2, 1)
    assert changes.content_changed_count == 1
    assert changes.metadata_only_changed_count == 1
    assert changes.governance_metadata_changed_count == 1
    assert changes.unassessed_missing_count == 0
    assert changes.category_count_deltas == {"cards": -1, "mortgages": 1}


def test_partial_snapshot_never_labels_missing_source_deleted() -> None:
    baseline = _snapshot(1, (_document("a"), _document("b")))
    candidate = _snapshot(2, (_document("a"),), authoritative=False)

    changes = assess_structural_drift(
        baseline,
        candidate,
        baseline_validation=_validation(1),
        candidate_validation=_validation(2),
        assessed_at=NOW,
    ).source_changes

    assert changes.deletion_detection_performed is False
    assert changes.deleted_count == 0
    assert changes.unassessed_missing_count == 1


def test_legacy_configuration_is_explicitly_incompatible() -> None:
    baseline = _snapshot(1, (_document("a"),), configuration={"bm25": {}})
    candidate = _snapshot(2, (_document("a"),))

    compatibility = assess_structural_drift(
        baseline,
        candidate,
        baseline_validation=_validation(1),
        candidate_validation=_validation(2),
        assessed_at=NOW,
    ).index_compatibility

    assert compatibility.baseline_schema_supported is False
    assert compatibility.comparison_supported is False
    assert compatibility.pii_detector_version_match is None


def test_current_but_different_governance_configuration_is_incompatible() -> None:
    baseline = _snapshot(1, (_document("a"),), configuration=_configuration(pii="v1"))
    candidate = _snapshot(2, (_document("a"),), configuration=_configuration(pii="v2"))

    compatibility = assess_structural_drift(
        baseline,
        candidate,
        baseline_validation=_validation(1),
        candidate_validation=_validation(2),
        assessed_at=NOW,
    ).index_compatibility

    assert compatibility.pii_detector_version_match is False
    assert compatibility.comparison_supported is False
    assert "PII detector version differs" in compatibility.incompatibilities


def _case(
    case_id: str,
    generation: int,
    *,
    hit: bool,
    expected_access: bool = True,
) -> RetrievalCaseResult:
    expected = (f"expected-{case_id}",)
    retrieved = (
        RetrievedIdentity(
            source_id=expected[0] if hit else "other",
            document_id="document",
            chunk_id="chunk",
        ),
    )
    return RetrievalCaseResult(
        case_id=case_id,
        generation=generation,
        qdrant_collection=f"managed_g{generation:06d}",
        identity_kind="source_id",
        expected_ids=expected,
        expected_access=expected_access,
        access_expectation_met=hit,
        retrieved_identities=retrieved,
        hit_at_k=hit,
        recall_at_k=float(hit),
        reciprocal_rank=float(hit),
        expected_rank=1 if hit else None,
        final_context_count=1,
    )


def _retrieval_report(
    generation: int,
    cases: tuple[RetrievalCaseResult, ...],
) -> RetrievalQualityReport:
    hits = sum(case.hit_at_k for case in cases)
    score = hits / len(cases)
    return RetrievalQualityReport(
        generated_at=NOW,
        generation=generation,
        index_state_id=UUID(int=generation),
        qdrant_collection=f"managed_g{generation:06d}",
        snapshot_hash=f"snapshot-{generation}",
        embedding_fingerprint="embedding-a",
        chunking_fingerprint="chunking-a",
        index_configuration_fingerprint="configuration-a",
        retrieval_settings=RetrievalRuntimeConfiguration(
            dense_candidates=2,
            sparse_candidates=2,
            fused_candidates=2,
            final_contexts=2,
            rrf_k=60,
            reranker_model="reranker-a",
            acl_filtering_enabled=True,
        ),
        case_count=len(cases),
        metrics=RetrievalQualityMetrics(
            k=2,
            recall_at_k=score,
            mrr=score,
            hit_rate=score,
            missed_case_count=len(cases) - hits,
        ),
        cases=cases,
    )


def test_retrieval_regression_reuses_quality_gate_and_reports_acl_failures() -> None:
    baseline = _retrieval_report(
        1,
        (_case("positive", 1, hit=True), _case("negative", 1, hit=True, expected_access=False)),
    )
    candidate = _retrieval_report(
        2,
        (_case("positive", 2, hit=False), _case("negative", 2, hit=False, expected_access=False)),
    )

    result = assess_retrieval_regression(
        baseline,
        candidate,
        thresholds=_thresholds(),
        comparison_supported=True,
    )

    assert result.status is AssessmentStatus.AVAILABLE
    assert result.threshold_result is not None
    assert result.threshold_result.status == "FAIL"
    assert result.regressed_case_ids == ("negative", "positive")
    assert result.newly_missed_expected_identities == (
        "expected-negative",
        "expected-positive",
    )
    assert result.acl_negative_failed_case_ids == ("negative",)


def _managed_answer_report(
    generation: int,
    *,
    score: float | None,
    judge_status: str = "COMPLETED",
    generation_model: str = "bedrock-model",
) -> ManagedRAGEvaluationReport:
    retrieval_case = _case("case-1", generation, hit=True)
    judge_metrics = {"correctness": score} if score is not None else {}
    return ManagedRAGEvaluationReport.model_validate(
        {
            "generated_at": NOW,
            "mode": "generated_answers",
            "corpus_key": "consumer-finance-managed",
            "snapshot_hash": f"snapshot-{generation}",
            "corpus_document_count": 1,
            "corpus_chunk_count": 1,
            "dataset_name": "reviewed-public",
            "dataset_version": "1.0",
            "dataset_format_version": "managed-rag-evaluation-dataset-v1",
            "dataset_privacy_classification": "PUBLIC_SYNTHETIC_REVIEWED",
            "case_count": 1,
            "generation": generation,
            "index_state_id": UUID(int=generation),
            "qdrant_collection": f"managed_g{generation:06d}",
            "embedding_fingerprint": "embedding-a",
            "chunking_fingerprint": "chunking-a",
            "index_configuration_fingerprint": "configuration-a",
            "models": {
                "embedding_model": "embedding-model",
                "reranker_model": "reranker-model",
                "generation_model": generation_model,
                "judge_model": "judge-model",
            },
            "retrieval_quality": {
                "k": 1,
                "recall_at_k": 1.0,
                "mrr": 1.0,
                "hit_rate": 1.0,
                "missed_case_count": 0,
            },
            "retrieval_cases": [retrieval_case.model_dump(mode="json")],
            "retrieval_latency": {"request_count": 1, "mean_total_ms": 1.0},
            "answer_quality": {
                "generation": {
                    "status": "COMPLETED",
                    "generated_case_count": 1,
                    "mean_latency_ms": 1.0,
                    "token_usage": {"available_case_count": 0},
                },
                "mlflow_genai": {
                    "status": judge_status,
                    "aggregate_metrics": judge_metrics,
                },
                "ragas": {"status": "NOT_REQUESTED"},
            },
            "answer_cases": [
                {
                    "case_id": "case-1",
                    "generation_status": "COMPLETED",
                    "answer_character_count": 10,
                    "citation_count": 1,
                    "cited_chunk_ids": ["chunk"],
                    "latency_ms": 1.0,
                    "token_usage": None,
                    "scoring": {
                        "mlflow_genai": judge_status,
                        "ragas": "NOT_REQUESTED",
                    },
                }
            ],
        }
    )


def test_answer_regression_requires_same_models_and_completed_judges() -> None:
    regressed = assess_answer_regression(
        _managed_answer_report(1, score=0.9),
        _managed_answer_report(2, score=0.7),
        thresholds=_thresholds(),
    )
    failed = assess_answer_regression(
        _managed_answer_report(1, score=0.9),
        _managed_answer_report(2, score=None, judge_status="FAILED"),
        thresholds=_thresholds(),
    )
    incompatible = assess_answer_regression(
        _managed_answer_report(1, score=0.9),
        _managed_answer_report(2, score=0.9, generation_model="different"),
        thresholds=_thresholds(),
    )

    assert regressed.status is AssessmentStatus.AVAILABLE
    assert regressed.violations
    assert failed.status is AssessmentStatus.FAILED
    assert incompatible.status is AssessmentStatus.INCOMPATIBLE


def _sample(
    moment: datetime,
    *,
    failure: bool = False,
    latency: float = 10,
) -> OperationalTelemetrySample:
    return OperationalTelemetrySample(
        timestamp=moment,
        generation=2,
        success=not failure,
        total_latency_ms=latency,
        dense_latency_ms=2,
        sparse_latency_ms=2,
        fusion_latency_ms=1,
        reranker_latency_ms=4,
        pii_latency_ms=1,
        dense_candidate_count=10,
        sparse_candidate_count=10,
        fused_candidate_count=8,
        final_context_count=4,
        redacted_context_count=1,
    )


def _window(
    start: datetime,
    samples: tuple[OperationalTelemetrySample, ...],
) -> OperationalTelemetryWindow:
    return OperationalTelemetryWindow(
        window_start=start,
        window_end=start + timedelta(hours=1),
        samples=samples,
    )


def test_operational_trends_require_equivalent_windows_and_minimum_samples() -> None:
    baseline_start = NOW - timedelta(hours=2)
    candidate_start = NOW - timedelta(hours=1)
    insufficient = assess_operational_trends(
        _window(baseline_start, (_sample(baseline_start),)),
        _window(candidate_start, (_sample(candidate_start),)),
        thresholds=_thresholds(samples=2),
    )
    available = assess_operational_trends(
        _window(
            baseline_start,
            (
                _sample(baseline_start),
                _sample(baseline_start + timedelta(minutes=1)),
            ),
        ),
        _window(
            candidate_start,
            (
                _sample(candidate_start, failure=True, latency=20),
                _sample(candidate_start + timedelta(minutes=1), latency=20),
            ),
        ),
        thresholds=_thresholds(samples=2),
    )

    assert insufficient.status is AssessmentStatus.INSUFFICIENT_DATA
    assert available.status is AssessmentStatus.AVAILABLE
    assert available.warnings
    assert available.candidate is not None
    assert available.candidate.mean_token_usage["total"] is None


def test_findings_separate_governance_missing_history_and_quality(tmp_path: Path) -> None:
    documents = (_document("a"),)
    baseline = _snapshot(1, documents)
    candidate = _snapshot(2, documents)
    structural = assess_structural_drift(
        baseline,
        candidate,
        baseline_validation=_validation(1, outcome="WARNING", physical=False),
        candidate_validation=_validation(2, outcome="FAIL", governance=True),
        assessed_at=NOW,
    )
    retrieval = assess_retrieval_regression(
        _retrieval_report(1, (_case("case", 1, hit=True),)),
        _retrieval_report(2, (_case("case", 2, hit=False),)),
        thresholds=_thresholds(),
        comparison_supported=True,
    )
    report = build_drift_report(
        corpus_key="consumer-finance-managed",
        baseline=baseline,
        candidate=candidate,
        evaluation=EvaluationIdentity(
            dataset_name="reviewed-public",
            dataset_version="1.0",
            dataset_format_version="managed-rag-evaluation-dataset-v1",
            dataset_privacy_classification="PUBLIC_SYNTHETIC_REVIEWED",
            embedding_model="embedding",
            reranker_model="reranker",
        ),
        thresholds=_thresholds(),
        structural=structural,
        retrieval=retrieval,
        answer_quality=assess_answer_regression(None, None, thresholds=_thresholds()),
        operational_trends=assess_operational_trends(None, None, thresholds=_thresholds()),
        generated_at=NOW,
    )
    path = write_drift_report(report, tmp_path / "drift.json")
    loaded = load_drift_report(path.read_bytes())
    categories = {item.category for item in loaded.findings}

    assert DriftFindingCategory.STRUCTURAL_FAILURE in categories
    assert DriftFindingCategory.GOVERNANCE_FAILURE in categories
    assert DriftFindingCategory.QUALITY_REGRESSION in categories
    assert DriftFindingCategory.INSUFFICIENT_DATA in categories
    serialized = path.read_text(encoding="utf-8")
    assert "raw question" not in serialized
    assert "access-group" not in serialized


def test_missing_historical_collection_is_incomplete_evidence_not_candidate_failure(
    tmp_path: Path,
) -> None:
    documents = (_document("a"),)
    baseline = _snapshot(1, documents)
    candidate = _snapshot(2, documents)
    structural = assess_structural_drift(
        baseline,
        candidate,
        baseline_validation=_validation(1, outcome="WARNING", physical=False),
        candidate_validation=_validation(2),
        assessed_at=NOW,
    )
    report = build_drift_report(
        corpus_key="consumer-finance-managed",
        baseline=baseline,
        candidate=candidate,
        evaluation=EvaluationIdentity(
            dataset_name="reviewed-public",
            dataset_version="1.0",
            dataset_format_version="managed-rag-evaluation-dataset-v1",
            dataset_privacy_classification="PUBLIC_SYNTHETIC_REVIEWED",
            embedding_model="embedding",
            reranker_model="reranker",
        ),
        thresholds=_thresholds(),
        structural=structural,
        retrieval=RetrievalRegressionAssessment(
            status=AssessmentStatus.UNAVAILABLE,
            reason="A required physical generation collection is unavailable.",
        ),
        answer_quality=assess_answer_regression(None, None, thresholds=_thresholds()),
        operational_trends=assess_operational_trends(None, None, thresholds=_thresholds()),
        generated_at=NOW,
    )

    categories = {finding.category for finding in report.findings}
    assert DriftFindingCategory.INSUFFICIENT_DATA in categories
    assert DriftFindingCategory.STRUCTURAL_FAILURE not in categories

    class UnavailableRun:
        active = False
        run_id = None

        def __getattr__(self, _name: str) -> object:
            raise AssertionError("unavailable MLflow must remain a no-op")

    report_path = write_drift_report(report, tmp_path / "missing-history.json")
    log_drift_report(report, report_path, UnavailableRun())  # type: ignore[arg-type]


def test_phase4_paths_contain_no_lifecycle_mutation_or_public_ui_controls() -> None:
    core = Path("src/consumer_finance_rag/drift.py").read_text(encoding="utf-8")
    cli = Path("scripts/assess_managed_drift.py").read_text(encoding="utf-8")
    public_ui = Path("app/streamlit_app.py").read_text(encoding="utf-8")
    source = core + cli

    for forbidden in (
        "publish(",
        "upsert_points(",
        "delete_generation_collection(",
        "transition_index_state(",
        "restore_generation(",
    ):
        assert forbidden not in source
    assert "assess_managed_drift" not in public_ui
    assert "Drift & Regression" not in public_ui
