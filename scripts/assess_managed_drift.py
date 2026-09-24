"""Assess managed generation drift and regression without lifecycle mutation."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path

from qdrant_client import QdrantClient
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from consumer_finance_rag.config import AppSettings, load_settings
from consumer_finance_rag.drift import (
    AnswerRegressionAssessment,
    AssessmentStatus,
    DriftThresholds,
    EvaluationIdentity,
    ManagedDriftAssessmentReport,
    OperationalTrendAssessment,
    RetrievalRegressionAssessment,
    assess_answer_regression,
    assess_operational_trends,
    assess_retrieval_regression,
    assess_structural_drift,
    build_drift_report,
    load_generation_snapshot,
    load_operational_window,
    log_drift_report,
    retrieval_report_from_managed_evaluation,
    summarize_physical_validation,
    write_drift_report,
)
from consumer_finance_rag.evaluation.managed_rag import (
    CollectingRetrievalTelemetrySink,
    ManagedRAGEvaluationDataset,
    ManagedRAGEvaluationReport,
    evaluate_managed_rag,
    load_managed_rag_dataset,
)
from consumer_finance_rag.evaluation.mlflow_genai import (
    JudgeConfigurationError,
    require_judge_model,
)
from consumer_finance_rag.evaluation.ragas_adapter import build_ragas_evaluator
from consumer_finance_rag.evaluation.tracking import evaluation_run
from consumer_finance_rag.factory import RetrievalRuntime, build_application_runtime
from consumer_finance_rag.lineage.database import (
    create_database_engine,
    require_database_url,
)
from consumer_finance_rag.lineage.index_operations import ManagedIndexOperationsService
from consumer_finance_rag.lineage.managed_retrieval import (
    ManagedRetrievalRuntime,
    build_managed_retrieval_runtime_for_generation,
)
from consumer_finance_rag.lineage.models import Corpus, IndexState
from consumer_finance_rag.lineage.repository import LineageRepository
from consumer_finance_rag.observability import NoOpTracer, TraceContentPolicy
from consumer_finance_rag.pipeline import RAGPipeline
from consumer_finance_rag.providers import (
    HuggingFaceEmbeddingProvider,
    HuggingFaceRerankerProvider,
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-generation", required=True, type=_positive_int)
    parser.add_argument(
        "--candidate-generation",
        type=_positive_int,
        default=None,
        help="Candidate generation (defaults to the current ACTIVE generation).",
    )
    parser.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Reviewed PUBLIC managed dataset (defaults to evaluation.managed_dataset_path).",
    )
    parser.add_argument(
        "--skip-retrieval-evaluation",
        action="store_true",
        help="Run metadata/validation assessment without loading retrieval models.",
    )
    parser.add_argument(
        "--generate-answers",
        action="store_true",
        help="Explicitly invoke Bedrock for baseline and candidate reviewed cases.",
    )
    parser.add_argument(
        "--with-mlflow-judges",
        action="store_true",
        help="Explicitly run existing MLflow GenAI judges for generated answers.",
    )
    parser.add_argument(
        "--with-ragas",
        action="store_true",
        help="Explicitly run the existing RAGAS evaluator for generated answers.",
    )
    parser.add_argument(
        "--deep-validation",
        action="store_true",
        help="Read and validate every point in baseline and candidate collections.",
    )
    parser.add_argument("--baseline-telemetry-window", type=Path, default=None)
    parser.add_argument("--candidate-telemetry-window", type=Path, default=None)
    parser.add_argument(
        "--log-mlflow",
        action="store_true",
        help="Log one metadata-only assessment run and report artifact.",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser


def _validate_modes(args: argparse.Namespace, settings: AppSettings) -> str | None:
    if settings.runtime.profile != "managed":
        raise SystemExit("Managed drift assessment requires RAG__RUNTIME__PROFILE=managed")
    if args.generate_answers and args.skip_retrieval_evaluation:
        raise SystemExit("--generate-answers cannot be combined with --skip-retrieval-evaluation")
    if (args.with_mlflow_judges or args.with_ragas) and not args.generate_answers:
        raise SystemExit("Judge flags require --generate-answers")
    if args.generate_answers and not settings.models.generation.model:
        raise SystemExit("--generate-answers requires an explicitly configured Bedrock model")
    if (args.with_mlflow_judges or args.log_mlflow) and not settings.mlflow.enabled:
        raise SystemExit("Requested MLflow work requires mlflow.enabled=true")
    if (args.baseline_telemetry_window is None) != (
        args.candidate_telemetry_window is None
    ):
        raise SystemExit("Both telemetry-window paths must be supplied together")
    if not (args.with_mlflow_judges or args.with_ragas):
        return None
    try:
        judge_model = require_judge_model(settings.mlflow)
    except JudgeConfigurationError as exc:
        raise SystemExit(str(exc)) from exc
    if args.with_ragas:
        build_ragas_evaluator(enabled=True, judge_model=judge_model, references={})
    return judge_model


def _active_generation(session: Session, corpus_key: str) -> int:
    corpus = session.scalar(select(Corpus).where(Corpus.corpus_key == corpus_key))
    if corpus is None:
        raise SystemExit(f"Managed corpus {corpus_key!r} does not exist")
    active = session.scalar(
        select(IndexState).where(
            IndexState.corpus_id == corpus.corpus_id,
            IndexState.status == "ACTIVE",
        )
    )
    if active is None:
        raise SystemExit("Managed corpus has no ACTIVE generation")
    return active.generation


def _point_count(
    client: QdrantClient, collection: str, *, present: bool
) -> int | None:
    if not present:
        return None
    try:
        return int(client.count(collection_name=collection, exact=True).count)
    except Exception:
        return None


def _shared_models(settings: AppSettings) -> tuple[
    HuggingFaceEmbeddingProvider,
    HuggingFaceRerankerProvider,
]:
    return (
        HuggingFaceEmbeddingProvider(
            settings.models.embedding.model,
            settings.models.embedding.device,
            settings.models.embedding.trust_remote_code,
            settings.models.embedding.query_prompt_name,
        ),
        HuggingFaceRerankerProvider(
            settings.models.reranker.model,
            settings.models.reranker.device,
            settings.models.reranker.trust_remote_code,
        ),
    )


def _runtime(
    settings: AppSettings,
    session: Session,
    client: QdrantClient,
    generation: int,
    embedding: HuggingFaceEmbeddingProvider,
    reranker: HuggingFaceRerankerProvider,
    telemetry: CollectingRetrievalTelemetrySink,
) -> ManagedRetrievalRuntime:
    return build_managed_retrieval_runtime_for_generation(
        settings,
        LineageRepository(session),
        generation,
        qdrant_client=client,
        embedding_provider=embedding,
        reranker_provider=reranker,
        tracer=NoOpTracer(),
        telemetry_sink=telemetry,
    )


def _pipeline(
    settings: AppSettings,
    runtime: ManagedRetrievalRuntime,
    telemetry: CollectingRetrievalTelemetrySink,
    *,
    content_tracing: bool,
) -> RAGPipeline:
    runtime_settings = settings
    if not content_tracing:
        runtime_settings = settings.model_copy(
            update={"mlflow": settings.mlflow.model_copy(update={"enabled": False})}
        )
    application = build_application_runtime(
        runtime_settings,
        retrieval_runtime=RetrievalRuntime(
            profile="managed",
            retrieval_service=runtime.retrieval_service,
            qdrant_collection=runtime.qdrant_collection,
            generation=runtime.generation,
            index_state_id=runtime.index_state_id,
        ),
        telemetry_sink=telemetry,
        trace_content_policy=(
            TraceContentPolicy.APPROVED_PUBLIC_EVALUATION
            if content_tracing
            else TraceContentPolicy.METADATA_ONLY
        ),
    )
    return application.pipeline


def _evaluate_pair(
    settings: AppSettings,
    dataset: ManagedRAGEvaluationDataset,
    baseline_runtime: ManagedRetrievalRuntime,
    candidate_runtime: ManagedRetrievalRuntime,
    baseline_telemetry: CollectingRetrievalTelemetrySink,
    candidate_telemetry: CollectingRetrievalTelemetrySink,
    *,
    generate: bool,
    with_mlflow_judges: bool,
    with_ragas: bool,
    judge_model: str | None,
) -> tuple[ManagedRAGEvaluationReport, ManagedRAGEvaluationReport]:
    baseline_pipeline = (
        _pipeline(
            settings,
            baseline_runtime,
            baseline_telemetry,
            content_tracing=with_mlflow_judges,
        )
        if generate
        else None
    )
    candidate_pipeline = (
        _pipeline(
            settings,
            candidate_runtime,
            candidate_telemetry,
            content_tracing=with_mlflow_judges,
        )
        if generate
        else None
    )
    baseline_report = evaluate_managed_rag(
        baseline_runtime,
        dataset,
        configured_corpus_key=settings.managed_ingestion.corpus_key,
        pipeline=baseline_pipeline,
        generation_model=settings.models.generation.model if generate else None,
        judge_model=judge_model,
        with_mlflow_judges=with_mlflow_judges,
        with_ragas=with_ragas,
        mlflow_settings=settings.mlflow,
        telemetry_sink=baseline_telemetry,
    )
    candidate_report = evaluate_managed_rag(
        candidate_runtime,
        dataset,
        configured_corpus_key=settings.managed_ingestion.corpus_key,
        pipeline=candidate_pipeline,
        generation_model=settings.models.generation.model if generate else None,
        judge_model=judge_model,
        with_mlflow_judges=with_mlflow_judges,
        with_ragas=with_ragas,
        mlflow_settings=settings.mlflow,
        telemetry_sink=candidate_telemetry,
    )
    return baseline_report, candidate_report


def _assessment_output(
    settings: AppSettings,
    baseline_generation: int,
    candidate_generation: int,
    requested: Path | None,
) -> Path:
    return requested or (
        settings.mlflow.report_directory
        / (
            f"managed_drift_g{baseline_generation:06d}_to_"
            f"g{candidate_generation:06d}.json"
        )
    )


def _print_report(report: ManagedDriftAssessmentReport, output: Path) -> None:
    print("Managed drift and regression assessment (READ ONLY)")
    print()
    print(f"Baseline: g{report.baseline.generation:06d}")
    print(f"Candidate: g{report.candidate.generation:06d}")
    print(
        f"Source changes: NEW={report.structural.source_changes.new_count} "
        f"CHANGED={report.structural.source_changes.changed_count} "
        f"UNCHANGED={report.structural.source_changes.unchanged_count} "
        f"DELETED={report.structural.source_changes.deleted_count}"
    )
    print(f"Retrieval assessment: {report.retrieval.status.value}")
    if report.retrieval.threshold_result is not None:
        print(f"Retrieval thresholds: {report.retrieval.threshold_result.status}")
    print(f"Answer-quality assessment: {report.answer_quality.status.value}")
    print(f"Operational trends: {report.operational_trends.status.value}")
    print("Findings:")
    for finding in report.findings:
        print(f"  {finding.category.value}: {finding.code}")
    print(f"JSON report: {output}")
    print("Lifecycle actions executed: NONE")


def _log_assessment(
    settings: AppSettings,
    report: ManagedDriftAssessmentReport,
    output: Path,
) -> str:
    """Attempt optional reporting without exposing backend error details."""

    try:
        with evaluation_run(
            settings.mlflow,
            run_name=(
                f"managed-drift-g{report.baseline.generation:06d}-"
                f"g{report.candidate.generation:06d}"
            ),
        ) as run:
            log_drift_report(report, output, run)
            if run.run_id is not None:
                return f"MLflow assessment run ID: {run.run_id}"
            return "MLflow assessment logging: UNAVAILABLE (no active run)"
    except Exception as exc:
        return f"MLflow assessment logging: UNAVAILABLE ({type(exc).__name__})"


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = load_settings(args.config)
    judge_model = _validate_modes(args, settings)
    require_database_url(settings.database)
    thresholds = DriftThresholds.from_settings(settings.drift_monitoring)
    dataset = load_managed_rag_dataset(
        args.dataset or settings.evaluation.managed_dataset_path
    )
    engine = create_database_engine(settings.database)
    sessions = sessionmaker(bind=engine, class_=Session, expire_on_commit=False)
    client = QdrantClient(url=settings.qdrant.url)
    try:
        with sessions() as session:
            candidate_generation = args.candidate_generation or _active_generation(
                session, settings.managed_ingestion.corpus_key
            )
            baseline_snapshot = load_generation_snapshot(
                session,
                corpus_key=settings.managed_ingestion.corpus_key,
                generation=args.baseline_generation,
            )
            candidate_snapshot = load_generation_snapshot(
                session,
                corpus_key=settings.managed_ingestion.corpus_key,
                generation=candidate_generation,
            )
        operations = ManagedIndexOperationsService(
            session_factory=sessions,
            qdrant_client=client,
            corpus_key=settings.managed_ingestion.corpus_key,
            settings=settings.managed_index,
        )
        baseline_validation_result = operations.validate_generation(
            args.baseline_generation, deep=bool(args.deep_validation)
        )
        candidate_validation_result = operations.validate_generation(
            candidate_generation, deep=bool(args.deep_validation)
        )
        baseline_validation = summarize_physical_validation(
            baseline_validation_result,
            membership_count=baseline_snapshot.membership_count,
            physical_point_count=_point_count(
                client,
                baseline_snapshot.identity.qdrant_collection,
                present=baseline_validation_result.physical_present,
            ),
        )
        candidate_validation = summarize_physical_validation(
            candidate_validation_result,
            membership_count=candidate_snapshot.membership_count,
            physical_point_count=_point_count(
                client,
                candidate_snapshot.identity.qdrant_collection,
                present=candidate_validation_result.physical_present,
            ),
        )
        structural = assess_structural_drift(
            baseline_snapshot,
            candidate_snapshot,
            baseline_validation=baseline_validation,
            candidate_validation=candidate_validation,
        )

        evaluation_identity = EvaluationIdentity(
            dataset_name=dataset.name,
            dataset_version=dataset.version,
            dataset_format_version=dataset.format_version,
            dataset_privacy_classification=dataset.privacy_classification,
            embedding_model=settings.models.embedding.model,
            reranker_model=settings.models.reranker.model,
            generation_model=(
                settings.models.generation.model if args.generate_answers else None
            ),
            judge_model=judge_model,
        )
        baseline_answer = None
        candidate_answer = None
        if args.skip_retrieval_evaluation:
            retrieval = RetrievalRegressionAssessment(
                status=AssessmentStatus.NOT_REQUESTED,
                reason="Retrieval evaluation was explicitly skipped.",
            )
        elif not structural.index_compatibility.comparison_supported:
            retrieval = RetrievalRegressionAssessment(
                status=AssessmentStatus.INCOMPATIBLE,
                reason="Generation index configurations are not safely comparable.",
            )
        elif not baseline_validation.physical_present or not candidate_validation.physical_present:
            retrieval = RetrievalRegressionAssessment(
                status=AssessmentStatus.UNAVAILABLE,
                reason="A required physical generation collection is unavailable.",
            )
        else:
            baseline_telemetry = CollectingRetrievalTelemetrySink()
            candidate_telemetry = CollectingRetrievalTelemetrySink()
            try:
                embedding, reranker = _shared_models(settings)
                with sessions() as session:
                    baseline_runtime = _runtime(
                        settings,
                        session,
                        client,
                        args.baseline_generation,
                        embedding,
                        reranker,
                        baseline_telemetry,
                    )
                    candidate_runtime = _runtime(
                        settings,
                        session,
                        client,
                        candidate_generation,
                        embedding,
                        reranker,
                        candidate_telemetry,
                    )
                baseline_evaluation, candidate_evaluation = _evaluate_pair(
                    settings,
                    dataset,
                    baseline_runtime,
                    candidate_runtime,
                    baseline_telemetry,
                    candidate_telemetry,
                    generate=bool(args.generate_answers),
                    with_mlflow_judges=bool(args.with_mlflow_judges),
                    with_ragas=bool(args.with_ragas),
                    judge_model=judge_model,
                )
                retrieval = assess_retrieval_regression(
                    retrieval_report_from_managed_evaluation(
                        baseline_evaluation, baseline_runtime
                    ),
                    retrieval_report_from_managed_evaluation(
                        candidate_evaluation, candidate_runtime
                    ),
                    thresholds=thresholds,
                    comparison_supported=True,
                )
                if args.generate_answers:
                    baseline_answer = baseline_evaluation
                    candidate_answer = candidate_evaluation
            except Exception as exc:
                retrieval = RetrievalRegressionAssessment(
                    status=AssessmentStatus.FAILED,
                    reason=f"Retrieval evaluation failed ({type(exc).__name__}).",
                )

        answer = assess_answer_regression(
            baseline_answer,
            candidate_answer,
            thresholds=thresholds,
        )
        if args.generate_answers and baseline_answer is None:
            answer = AnswerRegressionAssessment(
                status=(
                    AssessmentStatus.INCOMPATIBLE
                    if retrieval.status is AssessmentStatus.INCOMPATIBLE
                    else AssessmentStatus.FAILED
                ),
                reason=(
                    "Generated-answer comparison requires compatible generations."
                    if retrieval.status is AssessmentStatus.INCOMPATIBLE
                    else "Generated-answer comparison could not be completed."
                ),
            )
        operational: OperationalTrendAssessment
        if args.baseline_telemetry_window is None:
            operational = assess_operational_trends(None, None, thresholds=thresholds)
        else:
            operational = assess_operational_trends(
                load_operational_window(args.baseline_telemetry_window),
                load_operational_window(args.candidate_telemetry_window),
                thresholds=thresholds,
            )
        report = build_drift_report(
            corpus_key=settings.managed_ingestion.corpus_key,
            baseline=baseline_snapshot,
            candidate=candidate_snapshot,
            evaluation=evaluation_identity,
            thresholds=thresholds,
            structural=structural,
            retrieval=retrieval,
            answer_quality=answer,
            operational_trends=operational,
        )
        output = write_drift_report(
            report,
            _assessment_output(
                settings,
                args.baseline_generation,
                candidate_generation,
                args.output,
            ),
        )
        if args.log_mlflow:
            print(_log_assessment(settings, report, output))
        _print_report(report, output)
        return 0
    finally:
        with suppress(Exception):
            client.close()
        engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
