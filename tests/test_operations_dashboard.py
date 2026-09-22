from __future__ import annotations

import ast
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID

from consumer_finance_rag.config import MLflowSettings
from consumer_finance_rag.lineage.index_operations import (
    CleanupClassification,
    GenerationCleanupCandidate,
    GenerationCleanupPlan,
    ManagedGenerationStatus,
    ManagedGenerationValidationReport,
    ManagedGenerationValidationResult,
)
from consumer_finance_rag.operations_dashboard import (
    cleanup_plan_rows,
    collect_operations_snapshot,
    compare_quality_for_dashboard,
    comparison_summary,
    generation_history_rows,
    inspect_mlflow,
    load_quality_report_json,
    probe_component,
    quality_summary,
    restoration_provenance_rows,
    safe_identifier,
    validation_rows,
)
from consumer_finance_rag.retrieval_quality import (
    RetrievalCaseResult,
    RetrievalQualityMetrics,
    RetrievalQualityReport,
    RetrievalQualityThresholds,
    RetrievalRuntimeConfiguration,
    RetrievedIdentity,
)

NOW = datetime(2026, 9, 22, 5, 6, 7, tzinfo=UTC)
CORPUS = "consumer-finance-managed"


def generation_status(
    generation: int,
    status: str,
    *,
    physical: bool = True,
    restored_from: UUID | None = None,
    error_message: str | None = None,
) -> ManagedGenerationStatus:
    classification = {
        "ACTIVE": CleanupClassification.ACTIVE_PROTECTED,
        "RETIRED": CleanupClassification.RETAINED,
        "FAILED": CleanupClassification.ELIGIBLE_FAILED,
    }[status]
    return ManagedGenerationStatus(
        index_state_id=UUID(int=generation),
        generation=generation,
        status=status,
        qdrant_collection=f"consumer_finance_managed_g{generation:06d}",
        physical_present=physical,
        ingestion_run_id=UUID(int=100 + generation),
        document_count=10 + generation,
        chunk_count=20 + generation,
        embedding_dimension=384,
        created_at=NOW - timedelta(days=4 - generation),
        activated_at=NOW - timedelta(days=3 - generation)
        if status in {"ACTIVE", "RETIRED"}
        else None,
        retired_at=NOW - timedelta(days=1) if status == "RETIRED" else None,
        failed_at=NOW - timedelta(days=1) if status == "FAILED" else None,
        error_message=error_message,
        restored_from_index_state_id=restored_from,
        restoration_reason="raw operator reason secret=never-render",
        cleanup_classification=classification,
        cleanup_reason="raw planner reason password=never-render",
    )


def validation_report(*, deep: bool = False) -> ManagedGenerationValidationReport:
    return ManagedGenerationValidationReport(
        corpus_key=CORPUS,
        active_count=1,
        valid=False,
        issues=("expected exactly one ACTIVE generation, found 1",),
        generations=(
            ManagedGenerationValidationResult(
                index_state_id=UUID(int=3),
                generation=3,
                status="ACTIVE",
                qdrant_collection="consumer_finance_managed_g000003",
                physical_present=True,
                deep=deep,
                valid=False,
                issues=(
                    "collection is not readable: "
                    "postgresql://admin:password@internal/raw chunk text",
                ),
                notes=("physical collection cannot be validated without configuration",),
            ),
        ),
    )


def healthy_validation_report() -> ManagedGenerationValidationReport:
    return ManagedGenerationValidationReport(
        corpus_key=CORPUS,
        active_count=1,
        valid=True,
        issues=(),
        generations=(
            ManagedGenerationValidationResult(
                index_state_id=UUID(int=3),
                generation=3,
                status="ACTIVE",
                qdrant_collection="consumer_finance_managed_g000003",
                physical_present=True,
                deep=False,
                valid=True,
                issues=(),
                notes=(),
            ),
        ),
    )


def cleanup_plan() -> GenerationCleanupPlan:
    return GenerationCleanupPlan(
        corpus_key=CORPUS,
        managed_prefix="consumer_finance_managed",
        planned_at=NOW,
        retained_retired_generations=2,
        retired_cleanup_min_age_hours=24.0,
        candidates=(
            GenerationCleanupCandidate(
                index_state_id=UUID(int=3),
                generation=3,
                status="ACTIVE",
                qdrant_collection="consumer_finance_managed_g000003",
                physical_present=True,
                classification=CleanupClassification.ACTIVE_PROTECTED,
                reason="raw protected reason api_key=never-render",
            ),
            GenerationCleanupCandidate(
                index_state_id=UUID(int=2),
                generation=2,
                status="RETIRED",
                qdrant_collection="consumer_finance_managed_g000002",
                physical_present=True,
                classification=CleanupClassification.RETAINED,
                reason="raw retention reason secret=never-render",
            ),
        ),
    )


class FakeOperationsService:
    def __init__(self) -> None:
        self.status_calls = 0
        self.validate_calls: list[bool] = []
        self.cleanup_calls = 0
        self.write_calls = 0
        self.statuses = (
            generation_status(1, "FAILED", physical=False, error_message="password=secret"),
            generation_status(2, "RETIRED"),
            generation_status(3, "ACTIVE", restored_from=UUID(int=2)),
        )

    def status(self) -> tuple[ManagedGenerationStatus, ...]:
        self.status_calls += 1
        return self.statuses

    def validate(self, *, deep: bool = False) -> ManagedGenerationValidationReport:
        self.validate_calls.append(deep)
        return validation_report(deep=deep)

    def cleanup_plan(self) -> GenerationCleanupPlan:
        self.cleanup_calls += 1
        return cleanup_plan()

    def execute_cleanup(self, *_args: object, **_kwargs: object) -> None:
        self.write_calls += 1
        raise AssertionError("dashboard must not execute cleanup")


class HealthyOperationsService(FakeOperationsService):
    def validate(self, *, deep: bool = False) -> ManagedGenerationValidationReport:
        self.validate_calls.append(deep)
        return healthy_validation_report()


class PartiallyUnavailableOperationsService(FakeOperationsService):
    def status(self) -> tuple[ManagedGenerationStatus, ...]:
        raise ConnectionError("postgresql://admin:password@internal/raw text")

    def cleanup_plan(self) -> GenerationCleanupPlan:
        raise RuntimeError("api_key=never-render")

    def validate(self, *, deep: bool = False) -> ManagedGenerationValidationReport:
        self.validate_calls.append(deep)
        return healthy_validation_report()


def quality_report(generation: int, *, hit: bool) -> RetrievalQualityReport:
    collection = f"consumer_finance_managed_g{generation:06d}"
    case = RetrievalCaseResult(
        case_id="safe-case",
        generation=generation,
        qdrant_collection=collection,
        identity_kind="source_id",
        expected_ids=("safe-source",),
        expected_access=True,
        access_expectation_met=hit,
        retrieved_identities=(
            RetrievedIdentity(
                source_id="safe-source" if hit else "other-source",
                document_id="safe-document",
                chunk_id="safe-chunk",
            ),
        ),
        hit_at_k=hit,
        recall_at_k=float(hit),
        reciprocal_rank=float(hit),
        expected_rank=1 if hit else None,
        final_context_count=1,
    )
    return RetrievalQualityReport(
        generated_at=NOW,
        generation=generation,
        index_state_id=UUID(int=generation),
        qdrant_collection=collection,
        snapshot_hash=f"snapshot-{generation}",
        embedding_fingerprint=f"embedding-{generation}",
        chunking_fingerprint=f"chunking-{generation}",
        index_configuration_fingerprint=f"configuration-{generation}",
        retrieval_settings=RetrievalRuntimeConfiguration(
            dense_candidates=20,
            sparse_candidates=20,
            fused_candidates=12,
            final_contexts=5,
            rrf_k=60,
            reranker_model="safe-reranker",
            acl_filtering_enabled=True,
        ),
        case_count=1,
        metrics=RetrievalQualityMetrics(
            k=5,
            recall_at_k=float(hit),
            mrr=float(hit),
            hit_rate=float(hit),
            missed_case_count=0 if hit else 1,
        ),
        cases=(case,),
    )


def test_healthy_active_failed_retired_and_restoration_views_are_allowlisted() -> None:
    service = HealthyOperationsService()
    point_calls: list[str] = []

    def count_points(collection: str) -> int:
        point_calls.append(collection)
        return 23

    snapshot = collect_operations_snapshot(
        service,
        corpus_key=CORPUS,
        point_counter=count_points,
    )

    assert snapshot.active is not None and snapshot.active.generation == 3
    assert snapshot.active_point_count == 23
    assert snapshot.validation is not None and snapshot.validation.valid
    assert service.status_calls == service.cleanup_calls == 1
    assert service.validate_calls == [False]
    assert service.write_calls == 0
    assert point_calls == ["consumer_finance_managed_g000003"]

    history = generation_history_rows(snapshot.statuses)
    assert [row["status"] for row in history] == ["FAILED", "RETIRED", "★ ACTIVE"]
    assert history[0]["failure"] == "Recorded (details withheld)"
    assert history[2]["restored_from_index_state_id"] == str(UUID(int=2))
    provenance = restoration_provenance_rows(snapshot.statuses)
    assert provenance == (
        {
            "generation": 3,
            "index_state_id": str(UUID(int=3)),
            "restored_from_index_state_id": str(UUID(int=2)),
            "collection": "consumer_finance_managed_g000003",
        },
    )


def test_validation_and_cleanup_render_failed_invariants_and_protection() -> None:
    validation = validation_rows(validation_report(deep=True))
    cleanup = cleanup_plan_rows(cleanup_plan())

    assert validation[0]["mode"] == "deep"
    assert validation[0]["result"] == "FAIL"
    assert validation[0]["issues"] == "collection is not readable"
    assert cleanup[0]["classification"] == "ACTIVE_PROTECTED"
    assert cleanup[0]["protected"] is True
    assert cleanup[0]["reason"] == "Current ACTIVE generation is always protected."
    assert cleanup[1]["reason"] == (
        "Within the configured retired-generation retention set."
    )


def test_postgresql_and_qdrant_failures_are_nonfatal_and_secret_safe() -> None:
    def unavailable() -> None:
        raise ConnectionError(
            "postgresql://admin:password@db/internal?api_key=secret"
        )

    postgres = probe_component("PostgreSQL", unavailable)
    qdrant = probe_component("Qdrant", unavailable)

    assert not postgres.available
    assert not qdrant.available
    rendered = repr((postgres, qdrant))
    assert "ConnectionError" in rendered
    assert "postgresql://" not in rendered
    assert "password" not in rendered
    assert "api_key" not in rendered

    partial = collect_operations_snapshot(
        PartiallyUnavailableOperationsService(),
        corpus_key=CORPUS,
        point_counter=lambda _collection: 0,
    )
    assert partial.statuses == ()
    assert partial.validation is not None
    assert partial.cleanup_plan is None
    assert "ConnectionError" in (partial.status_error or "")
    assert "RuntimeError" in (partial.cleanup_error or "")
    assert "password" not in repr(partial)
    assert "api_key" not in repr(partial)


def test_quality_report_loading_summary_and_existing_comparison_helpers() -> None:
    baseline = quality_report(3, hit=False)
    candidate = quality_report(4, hit=True)
    payload = json.loads(candidate.model_dump_json())
    payload["raw_query"] = "alice@example.test secret query"
    payload["cases"][0]["raw_chunk_text"] = "raw regulated context"
    loaded = load_quality_report_json(json.dumps(payload))

    summary = quality_summary(loaded)
    assert summary["generation"] == 4
    assert summary["recall_at_k"] == 1.0
    assert summary["mrr"] == 1.0
    assert summary["hit_rate"] == 1.0
    assert summary["missed_case_count"] == 0
    assert "alice@example.test" not in repr(summary)
    assert "raw regulated context" not in repr(summary)

    view = compare_quality_for_dashboard(
        baseline,
        loaded,
        RetrievalQualityThresholds(max_recall_drop=0.0, max_mrr_drop=0.0),
    )
    compared = comparison_summary(view)
    assert compared["recall_delta"] == 1.0
    assert compared["mrr_delta"] == 1.0
    assert compared["newly_recovered_case_count"] == 1
    assert compared["threshold_result"] == "PASS"
    assert view.comparison.cases_improved == ("safe-case",)
    assert view.comparison.newly_recovered_cases == ("safe-case",)


class UnavailableMLflowClient:
    def get_experiment_by_name(self, name: str) -> object:
        raise ConnectionError(f"secret={name}")


class AvailableMLflowClient:
    starts = 0

    def get_experiment_by_name(self, name: str) -> object:
        assert name == "consumer-finance-rag-evaluation"
        return type("Experiment", (), {"experiment_id": "42"})()


def test_mlflow_is_optional_read_only_and_unavailable_is_nonfatal() -> None:
    settings = MLflowSettings(
        enabled=True,
        tracking_uri="http://localhost:5000",
        evaluation_experiment_name="consumer-finance-rag-evaluation",
    )
    unavailable = inspect_mlflow(
        settings,
        client_factory=lambda _uri: UnavailableMLflowClient(),
    )
    available = inspect_mlflow(
        settings,
        client_factory=lambda _uri: AvailableMLflowClient(),
    )
    disabled = inspect_mlflow(settings.model_copy(update={"enabled": False}))

    assert not unavailable.available
    assert "secret" not in unavailable.message
    assert available.available
    assert available.link == "http://localhost:5000/#/experiments/42"
    assert not disabled.configured
    assert AvailableMLflowClient.starts == 0


def test_rendered_operational_data_contains_no_secrets_or_raw_text() -> None:
    service = FakeOperationsService()
    snapshot = collect_operations_snapshot(
        service,
        corpus_key=CORPUS,
        point_counter=lambda _collection: 23,
        deep_validation=True,
    )
    rendered = repr(
        (
            generation_history_rows(snapshot.statuses),
            validation_rows(snapshot.validation) if snapshot.validation else (),
            cleanup_plan_rows(snapshot.cleanup_plan) if snapshot.cleanup_plan else (),
            restoration_provenance_rows(snapshot.statuses),
        )
    )

    for forbidden in (
        "password=secret",
        "postgresql://",
        "raw chunk text",
        "raw operator reason",
        "raw planner reason",
        "api_key=never-render",
    ):
        assert forbidden not in rendered
    assert safe_identifier("safe-case:001") == "safe-case:001"
    assert safe_identifier("alice@example.test raw query") == (
        "<withheld-invalid-identifier>"
    )


def test_dashboard_imports_no_models_generation_or_mutation_paths() -> None:
    forbidden_imports = {
        "consumer_finance_rag.bedrock",
        "consumer_finance_rag.factory",
        "consumer_finance_rag.pipeline",
        "consumer_finance_rag.providers",
    }
    for path in (
        Path("app/operations_dashboard.py"),
        Path("src/consumer_finance_rag/operations_dashboard.py"),
    ):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        imported = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        assert not (imported & forbidden_imports)
        assert "execute_cleanup(" not in source
        assert "delete_generation_collection(" not in source
        assert "start_run(" not in source
        assert ".upsert(" not in source
        assert ".delete(" not in source
        assert ".create_collection(" not in source
        assert ".commit(" not in source
        assert "session.begin(" not in source
