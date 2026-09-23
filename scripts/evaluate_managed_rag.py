"""Evaluate retrieval and optional generated answers from one pinned managed generation."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from sqlalchemy.orm import Session

from consumer_finance_rag.config import AppSettings, load_settings
from consumer_finance_rag.evaluation.managed_rag import (
    CollectingRetrievalTelemetrySink,
    ManagedRAGEvaluationCase,
    ManagedRAGEvaluationDataset,
    evaluate_managed_rag,
    load_managed_rag_dataset,
    log_managed_rag_report,
    validate_managed_dataset_compatibility,
    write_managed_rag_report,
)
from consumer_finance_rag.evaluation.mlflow_genai import (
    JudgeConfigurationError,
    require_judge_model,
)
from consumer_finance_rag.evaluation.ragas_adapter import build_ragas_evaluator
from consumer_finance_rag.evaluation.tracking import evaluation_run
from consumer_finance_rag.factory import (
    RetrievalRuntime,
    build_application_runtime,
)
from consumer_finance_rag.lineage.database import (
    create_database_engine,
    require_database_url,
)
from consumer_finance_rag.lineage.managed_retrieval import (
    ManagedRetrievalRuntime,
    build_managed_retrieval_runtime,
)
from consumer_finance_rag.lineage.repository import LineageRepository
from consumer_finance_rag.observability import NoOpTracer


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--current",
        action="store_true",
        help="Pin and evaluate the current ACTIVE managed generation.",
    )
    parser.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Reviewed PUBLIC managed dataset (defaults to evaluation.managed_dataset_path).",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Explicitly generate answers through the configured Bedrock model.",
    )
    parser.add_argument("--limit", type=_positive_int, default=None)
    parser.add_argument(
        "--with-mlflow-judges",
        action="store_true",
        help="Explicitly run the existing MLflow GenAI judge batch.",
    )
    parser.add_argument(
        "--with-ragas",
        action="store_true",
        help="Explicitly run the existing RAGAS 0.4 evaluator.",
    )
    parser.add_argument(
        "--log-mlflow",
        action="store_true",
        help="Log the safe report and aggregate metrics in one MLflow evaluation run.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Safe JSON report path (defaults under mlflow.report_directory).",
    )
    return parser


def _build_current_runtime(
    settings: AppSettings,
    telemetry_sink: CollectingRetrievalTelemetrySink,
) -> ManagedRetrievalRuntime:
    require_database_url(settings.database)
    engine = create_database_engine(settings.database)
    try:
        with Session(engine) as session:
            return build_managed_retrieval_runtime(
                settings,
                LineageRepository(session),
                tracer=NoOpTracer(),
                telemetry_sink=telemetry_sink,
            )
    finally:
        engine.dispose()


def _validate_requested_modes(
    args: argparse.Namespace,
    settings: AppSettings,
) -> str | None:
    if (args.with_mlflow_judges or args.with_ragas) and not args.generate:
        raise SystemExit("--with-mlflow-judges/--with-ragas require --generate")
    if args.generate and not settings.models.generation.model:
        raise SystemExit(
            "--generate requires BEDROCK_GENERATION_MODEL or "
            "RAG__MODELS__GENERATION__MODEL"
        )
    if (args.with_mlflow_judges or args.log_mlflow) and not settings.mlflow.enabled:
        raise SystemExit("Requested MLflow evaluation requires mlflow.enabled=true")
    if not (args.with_mlflow_judges or args.with_ragas):
        return None
    try:
        judge_model = require_judge_model(settings.mlflow)
    except JudgeConfigurationError as exc:
        raise SystemExit(str(exc)) from exc
    if args.with_ragas:
        # Constructor validation occurs before runtime/model initialization.
        build_ragas_evaluator(enabled=True, judge_model=judge_model, references={})
    return judge_model


def _content_tracking_settings(
    settings: AppSettings,
    *,
    enabled: bool,
) -> AppSettings:
    if enabled:
        return settings
    return settings.model_copy(
        update={"mlflow": settings.mlflow.model_copy(update={"enabled": False})}
    )


def _last_trace_id() -> str | None:
    try:
        import mlflow

        return mlflow.get_last_active_trace_id()
    except Exception:
        return None


def _report_path(
    settings: AppSettings,
    runtime: ManagedRetrievalRuntime,
    requested: Path | None,
) -> Path:
    return requested or (
        settings.mlflow.report_directory
        / f"managed_rag_g{runtime.generation:06d}.json"
    )


def _selected_cases(
    dataset: ManagedRAGEvaluationDataset,
    limit: int | None,
) -> tuple[ManagedRAGEvaluationCase, ...]:
    return dataset.limited_cases(limit)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = load_settings(args.config)
    judge_model = _validate_requested_modes(args, settings)
    dataset = load_managed_rag_dataset(
        args.dataset or settings.evaluation.managed_dataset_path
    )
    selected_cases = _selected_cases(dataset, args.limit)

    telemetry = CollectingRetrievalTelemetrySink()
    runtime = _build_current_runtime(settings, telemetry)
    validate_managed_dataset_compatibility(
        dataset,
        runtime,
        configured_corpus_key=settings.managed_ingestion.corpus_key,
        cases=selected_cases,
    )

    pipeline = None
    trace_id_provider = None
    if args.generate:
        content_tracking = bool(args.log_mlflow or args.with_mlflow_judges)
        application = build_application_runtime(
            _content_tracking_settings(settings, enabled=content_tracking),
            retrieval_runtime=RetrievalRuntime(
                profile="managed",
                retrieval_service=runtime.retrieval_service,
                qdrant_collection=runtime.qdrant_collection,
                generation=runtime.generation,
                index_state_id=runtime.index_state_id,
            ),
            telemetry_sink=telemetry,
        )
        pipeline = application.pipeline
        if content_tracking:
            trace_id_provider = _last_trace_id

    report = evaluate_managed_rag(
        runtime,
        dataset,
        configured_corpus_key=settings.managed_ingestion.corpus_key,
        selected_cases=selected_cases,
        pipeline=pipeline,
        generation_model=(settings.models.generation.model if args.generate else None),
        judge_model=judge_model,
        with_mlflow_judges=bool(args.with_mlflow_judges),
        with_ragas=bool(args.with_ragas),
        mlflow_settings=settings.mlflow,
        telemetry_sink=telemetry,
        trace_id_provider=trace_id_provider,
    )
    output = write_managed_rag_report(report, _report_path(settings, runtime, args.output))

    logged_run_id: str | None = None
    if args.log_mlflow:
        judge_run_id = report.answer_quality.mlflow_genai.run_id
        with evaluation_run(
            settings.mlflow,
            run_name=f"managed-rag-g{runtime.generation:06d}",
            existing_run_id=judge_run_id,
        ) as run:
            log_managed_rag_report(report, output, run)
            logged_run_id = run.run_id

    print("Managed RAG evaluation")
    print()
    print(f"Mode: {report.mode}")
    print(f"Generation: g{report.generation:06d}")
    print(f"Index state: {report.index_state_id}")
    print(f"Collection: {report.qdrant_collection}")
    print(f"Dataset: {report.dataset_name} v{report.dataset_version}")
    print(f"Cases: {report.case_count}")
    print(f"Recall@{report.retrieval_quality.k}: {report.retrieval_quality.recall_at_k:.4f}")
    print(f"MRR: {report.retrieval_quality.mrr:.4f}")
    print(f"Hit rate: {report.retrieval_quality.hit_rate:.4f}")
    print(f"Generation: {report.answer_quality.generation.status}")
    print(f"MLflow GenAI judges: {report.answer_quality.mlflow_genai.status}")
    print(f"RAGAS: {report.answer_quality.ragas.status}")
    print(f"JSON report: {output}")
    if logged_run_id is not None:
        print(f"MLflow run ID: {logged_run_id}")

    requested_failure = (
        args.with_mlflow_judges
        and report.answer_quality.mlflow_genai.status != "COMPLETED"
    ) or (args.with_ragas and report.answer_quality.ragas.status != "COMPLETED")
    return 3 if requested_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
