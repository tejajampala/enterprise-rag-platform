"""Evaluate the actual RAG pipeline; paid judge stages are explicit opt-ins."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean
from typing import Any

from consumer_finance_rag import __version__
from consumer_finance_rag.acquisition import JsonFixtureSource
from consumer_finance_rag.chunking import ParagraphChunker
from consumer_finance_rag.config import AppSettings, load_settings
from consumer_finance_rag.evaluation.dataset import load_evaluation_set, validate_evaluation_set
from consumer_finance_rag.evaluation.mlflow_genai import run_mlflow_judges
from consumer_finance_rag.evaluation.models import (
    JudgeMetricValue,
    RAGCaseResult,
    RAGEvaluationReport,
)
from consumer_finance_rag.evaluation.ragas_adapter import build_ragas_evaluator
from consumer_finance_rag.evaluation.reporting import write_rag_report, write_retrieval_report
from consumer_finance_rag.evaluation.retrieval_runner import evaluate_retrieval
from consumer_finance_rag.evaluation.tracking import (
    evaluation_run,
    flatten_group_metrics,
    flatten_stage_metrics,
    git_commit_sha,
)
from consumer_finance_rag.factory import build_pipeline, build_retrieval_service
from consumer_finance_rag.normalization import normalize_records
from consumer_finance_rag.observability import NoOpTracer, bedrock_token_usage


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--with-mlflow-judges", action="store_true")
    parser.add_argument("--with-ragas", action="store_true")
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--output-directory", type=Path, default=None)
    return parser


def _parameters(settings: AppSettings, dataset_name: str, dataset_version: str) -> dict[str, Any]:
    return {
        "app.version": __version__,
        "git.commit": git_commit_sha(),
        "evaluation.dataset_name": dataset_name,
        "evaluation.dataset_version": dataset_version,
        "embedding.provider": settings.models.embedding.provider,
        "embedding.model": settings.models.embedding.model,
        "reranker.provider": settings.models.reranker.provider,
        "reranker.model": settings.models.reranker.model,
        "generation.provider": settings.models.generation.provider,
        "generation.model": settings.models.generation.model,
        "judge.model": settings.mlflow.judge_model,
        "qdrant.collection": settings.qdrant.collection_name,
        "qdrant.hnsw_m": settings.qdrant.hnsw_m,
        "qdrant.hnsw_ef_construct": settings.qdrant.hnsw_ef_construct,
        "retrieval.dense_candidates": settings.retrieval.dense_candidates,
        "retrieval.sparse_candidates": settings.retrieval.sparse_candidates,
        "retrieval.fused_candidates": settings.retrieval.fused_candidates,
        "retrieval.final_contexts": settings.retrieval.final_contexts,
        "retrieval.rrf_k": settings.retrieval.rrf_k,
    }


def _aggregate_case_metrics(cases: Sequence[RAGCaseResult], attribute: str) -> dict[str, float]:
    values: dict[str, list[float]] = {}
    for case in cases:
        metrics: dict[str, JudgeMetricValue] = getattr(case, attribute)
        for name, metric in metrics.items():
            if isinstance(metric.value, int | float) and not isinstance(metric.value, bool):
                values.setdefault(name, []).append(float(metric.value))
    return {name: fmean(items) for name, items in values.items() if items}


def _trace_operations(trace_ids: Sequence[str]) -> dict[str, int | float | None]:
    operations: dict[str, int | float | None] = {
        "trace_count": len(trace_ids),
        "mean_latency_ms": None,
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
        "cost_usd": None,
    }
    if not trace_ids:
        return operations
    try:
        import mlflow

        traces = [mlflow.get_trace(trace_id, flush=True) for trace_id in trace_ids]
        available = [trace for trace in traces if trace is not None]
        durations = [trace.info.execution_duration for trace in available]
        known_durations = [duration for duration in durations if duration is not None]
        if known_durations:
            operations["mean_latency_ms"] = fmean(known_durations)
        costs = [trace.info.cost for trace in available if trace.info.cost is not None]
        if costs:
            operations["cost_usd"] = float(sum(costs))
    except Exception:
        # The report remains valid when an asynchronous trace is not yet queryable.
        pass
    return operations


def _token_totals(results: Sequence[RAGCaseResult]) -> dict[str, int | None]:
    keys = ("input_tokens", "output_tokens", "total_tokens")
    if not results or any(result.token_usage is None for result in results):
        return {key: None for key in keys}
    return {
        key: sum(result.token_usage[key] for result in results if result.token_usage is not None)
        for key in keys
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")
    if args.retrieval_only and (args.with_mlflow_judges or args.with_ragas):
        raise SystemExit("--retrieval-only cannot be combined with paid judge switches")

    settings = load_settings(args.config)
    if args.no_mlflow:
        settings = settings.model_copy(
            update={"mlflow": settings.mlflow.model_copy(update={"enabled": False})}
        )
    dataset = load_evaluation_set(args.dataset or settings.evaluation.dataset_path)
    if args.limit:
        dataset = dataset.model_copy(update={"cases": dataset.cases[: args.limit]})
    source = JsonFixtureSource("data/fixtures/cfpb_consumer_finance.json")
    documents = normalize_records(source.acquire())
    chunker = ParagraphChunker(settings.chunking.chunk_size, settings.chunking.overlap)
    chunks = [chunk for document in documents for chunk in chunker.chunk(document)]
    validate_evaluation_set(dataset, documents=documents, chunks=chunks)
    output_directory = args.output_directory or settings.mlflow.report_directory
    parameters = _parameters(settings, dataset.name, dataset.version)

    if args.retrieval_only:
        service = build_retrieval_service(settings, tracer=NoOpTracer())
        parameters["bm25.k1"] = service.sparse_encoder.state.k1
        parameters["bm25.b"] = service.sparse_encoder.state.b
        retrieval_report = evaluate_retrieval(
            service,
            dataset,
            k_values=settings.evaluation.k_values,
            parameters=parameters,
        )
        paths = write_retrieval_report(retrieval_report, output_directory)
        with evaluation_run(settings.mlflow, run_name="rag-retrieval-only") as run:
            run.log_parameters(parameters)
            metrics = flatten_stage_metrics(retrieval_report.aggregate)
            metrics.update(flatten_group_metrics(retrieval_report.per_category, prefix="category"))
            metrics.update(
                flatten_group_metrics(retrieval_report.per_difficulty, prefix="difficulty")
            )
            run.log_metrics(metrics)
            run.log_artifacts(list(paths))
        print(f"Retrieval-only evaluation complete: {paths[0]}")
        return 0

    pipeline = build_pipeline(settings)
    parameters["bm25.k1"] = pipeline.retrieval.sparse_encoder.state.k1
    parameters["bm25.b"] = pipeline.retrieval.sparse_encoder.state.b
    retrieval_report = evaluate_retrieval(
        pipeline.retrieval,
        dataset,
        k_values=settings.evaluation.k_values,
        parameters=parameters,
    )
    retrieval_paths = write_retrieval_report(retrieval_report, output_directory)
    retrieval_metrics = flatten_stage_metrics(retrieval_report.aggregate)
    generated: list[RAGCaseResult] = []
    trace_ids: list[str] = []
    for case in dataset.cases:
        response = pipeline.answer(case.question)
        trace_id: str | None = None
        if settings.mlflow.enabled:
            try:
                import mlflow

                trace_id = mlflow.get_last_active_trace_id()
            except Exception:
                trace_id = None
        if trace_id:
            trace_ids.append(trace_id)
        generated.append(
            RAGCaseResult(
                case_id=case.case_id,
                question=case.question,
                answer=response.answer,
                contexts=[item.chunk.text for item in response.retrieved_chunks],
                chunk_ids=[item.chunk.chunk_id for item in response.retrieved_chunks],
                token_usage=bedrock_token_usage(response.generation_metadata),
                trace_id=trace_id,
            )
        )

    if args.with_mlflow_judges:
        judge_results = run_mlflow_judges(dataset.cases, generated, settings.mlflow)
        generated = [
            item.model_copy(update={"mlflow_judges": judge_results.by_case.get(item.case_id, {})})
            for item in generated
        ]
    if args.with_ragas:
        references = {case.case_id: case.reference_answer for case in dataset.cases}
        ragas = build_ragas_evaluator(
            enabled=True,
            judge_model=settings.mlflow.judge_model,
            references=references,
        )
        ragas_results = ragas.evaluate(generated)
        generated = [
            item.model_copy(update={"ragas": ragas_results.get(item.case_id, {})})
            for item in generated
        ]

    mlflow_metrics = _aggregate_case_metrics(generated, "mlflow_judges")
    ragas_metrics = _aggregate_case_metrics(generated, "ragas")
    rag_metrics: dict[str, float | None] = {
        "correctness": mlflow_metrics.get("correctness"),
        "relevance": mlflow_metrics.get("relevance_to_query"),
        "safety": mlflow_metrics.get("safety"),
        "retrieval_groundedness": mlflow_metrics.get("retrieval_groundedness"),
        "faithfulness": ragas_metrics.get("faithfulness"),
        "context_precision": ragas_metrics.get("context_precision"),
        "context_recall": ragas_metrics.get("context_recall"),
        "response_groundedness": ragas_metrics.get("response_groundedness"),
    }
    operations = _trace_operations(trace_ids)
    operations.update(_token_totals(generated))
    report = RAGEvaluationReport(
        created_at=datetime.now(UTC).isoformat(),
        dataset_name=dataset.name,
        dataset_version=dataset.version,
        case_count=len(generated),
        retrieval=retrieval_metrics,
        rag=rag_metrics,
        operations=operations,
        cases=generated,
    )
    report_path = write_rag_report(report, output_directory)
    logged_metrics = {
        f"rag.{name}": value for name, value in rag_metrics.items() if value is not None
    }
    logged_metrics.update(retrieval_metrics)
    logged_metrics.update(flatten_group_metrics(retrieval_report.per_category, prefix="category"))
    logged_metrics.update(
        flatten_group_metrics(retrieval_report.per_difficulty, prefix="difficulty")
    )
    with evaluation_run(settings.mlflow, run_name="rag-evaluation") as run:
        run.log_parameters(parameters)
        run.log_metrics(logged_metrics)
        run.log_artifacts([report_path, *retrieval_paths])
    print(f"Evaluated {len(generated)} generated answers: {report_path}")
    if not args.with_mlflow_judges and not args.with_ragas:
        print("Paid judges: not run (enable explicitly with --with-mlflow-judges/--with-ragas)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
