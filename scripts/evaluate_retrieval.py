"""Evaluate dense, BM25, RRF, and reranked production retrieval stages."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from consumer_finance_rag import __version__
from consumer_finance_rag.config import AppSettings, load_settings
from consumer_finance_rag.corpus import CorpusSnapshot, load_corpus_snapshot
from consumer_finance_rag.evaluation.dataset import load_evaluation_set, validate_evaluation_set
from consumer_finance_rag.evaluation.models import EvaluationCase
from consumer_finance_rag.evaluation.reporting import write_retrieval_report
from consumer_finance_rag.evaluation.retrieval_runner import evaluate_retrieval
from consumer_finance_rag.evaluation.tracking import (
    evaluation_run,
    flatten_group_metrics,
    flatten_stage_metrics,
    git_commit_sha,
)
from consumer_finance_rag.factory import build_retrieval_service, warm_up_retrieval_service
from consumer_finance_rag.observability import NoOpTracer


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _k_values(value: str) -> list[int]:
    try:
        values = [_positive_int(item.strip()) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("K values must be comma-separated integers") from error
    if not values:
        raise argparse.ArgumentTypeError("at least one K value is required")
    return sorted(set(values))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--k", type=_k_values, default=None, help="Comma-separated K values.")
    parser.add_argument("--dense-candidates", type=_positive_int)
    parser.add_argument("--sparse-candidates", type=_positive_int)
    parser.add_argument("--fused-candidates", type=_positive_int)
    parser.add_argument("--final-contexts", type=_positive_int)
    parser.add_argument("--rrf-k", type=_positive_int)
    parser.add_argument("--output-directory", type=Path, default=None)
    parser.add_argument(
        "--max-cases",
        type=_positive_int,
        default=None,
        help="Evaluate the first N seed cases through the unchanged production path.",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Run deterministic evaluation without contacting the MLflow server.",
    )
    return parser


def _with_overrides(settings: AppSettings, args: argparse.Namespace) -> AppSettings:
    retrieval_updates = {
        name: value
        for name, value in {
            "dense_candidates": args.dense_candidates,
            "sparse_candidates": args.sparse_candidates,
            "fused_candidates": args.fused_candidates,
            "final_contexts": args.final_contexts,
            "rrf_k": args.rrf_k,
        }.items()
        if value is not None
    }
    retrieval = settings.retrieval.model_copy(update=retrieval_updates)
    mlflow = settings.mlflow.model_copy(update={"enabled": not args.no_mlflow})
    return settings.model_copy(update={"retrieval": retrieval, "mlflow": mlflow})


def _parameters(
    settings: AppSettings,
    dataset_name: str,
    dataset_version: str,
    corpus: CorpusSnapshot,
) -> dict[str, str | int | float | bool | None]:
    return {
        "embedding.provider": settings.models.embedding.provider,
        "embedding.model": settings.models.embedding.model,
        "reranker.provider": settings.models.reranker.provider,
        "reranker.model": settings.models.reranker.model,
        "qdrant.collection": settings.qdrant.collection_name,
        "retrieval.dense_candidates": settings.retrieval.dense_candidates,
        "retrieval.sparse_candidates": settings.retrieval.sparse_candidates,
        "retrieval.fused_candidates": settings.retrieval.fused_candidates,
        "retrieval.final_contexts": settings.retrieval.final_contexts,
        "retrieval.rrf_k": settings.retrieval.rrf_k,
        "qdrant.hnsw_m": settings.qdrant.hnsw_m,
        "qdrant.hnsw_ef_construct": settings.qdrant.hnsw_ef_construct,
        "evaluation.dataset_name": dataset_name,
        "evaluation.dataset_version": dataset_version,
        "corpus.document_count": len(corpus.documents),
        "corpus.chunk_count": len(corpus.chunks),
        "corpus.fixture_document_count": corpus.fixture_document_count,
        "corpus.expanded_document_count": corpus.expanded_document_count,
        "git.commit": git_commit_sha(),
        "app.version": __version__,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = _with_overrides(load_settings(args.config), args)
    dataset_path = args.dataset or settings.evaluation.dataset_path
    dataset = load_evaluation_set(dataset_path)
    if args.max_cases is not None:
        dataset = dataset.model_copy(update={"cases": dataset.cases[: args.max_cases]})

    corpus = load_corpus_snapshot(settings, require_expanded=True)
    validate_evaluation_set(dataset, documents=corpus.documents, chunks=corpus.chunks)
    print(
        f"Corpus: {len(corpus.documents)} documents / {len(corpus.chunks)} chunks "
        f"({corpus.fixture_document_count} fixture, "
        f"{corpus.expanded_document_count} downloaded).",
        flush=True,
    )
    print(f"Category distribution: {corpus.category_distribution}", flush=True)

    service = build_retrieval_service(settings, tracer=NoOpTracer())
    warm_up_retrieval_service(service, progress=lambda message: print(message, flush=True))
    parameters = _parameters(settings, dataset.name, dataset.version, corpus)
    parameters["bm25.k1"] = service.sparse_encoder.state.k1
    parameters["bm25.b"] = service.sparse_encoder.state.b
    for category, count in corpus.category_distribution.items():
        parameters[f"corpus.category.{category}"] = count

    def show_progress(position: int, total: int, case: EvaluationCase) -> None:
        print(f"Evaluating case {position}/{total}: {case.case_id}", flush=True)

    report = evaluate_retrieval(
        service,
        dataset,
        k_values=args.k or settings.evaluation.k_values,
        parameters=parameters,
        progress=show_progress,
    )
    output_directory = args.output_directory or settings.mlflow.report_directory
    json_path, csv_path = write_retrieval_report(report, output_directory)

    with evaluation_run(settings.mlflow, run_name="retrieval-evaluation") as run:
        run.log_parameters(parameters)
        metrics = flatten_stage_metrics(report.aggregate)
        metrics.update(flatten_group_metrics(report.per_category, prefix="category"))
        metrics.update(flatten_group_metrics(report.per_difficulty, prefix="difficulty"))
        run.log_metrics(metrics)
        run.log_artifacts([json_path, csv_path, dataset_path])

    print(f"Evaluated {report.case_count} seed cases.")
    print(f"JSON report: {json_path}")
    print(f"Per-case CSV: {csv_path}")
    if not settings.mlflow.enabled:
        print("MLflow logging: disabled by --no-mlflow")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
