"""Evaluate deterministic retrieval quality for the current managed generation."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from sqlalchemy.orm import Session

from consumer_finance_rag.config import load_settings
from consumer_finance_rag.evaluation.dataset import load_evaluation_set
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
from consumer_finance_rag.retrieval_quality import (
    RetrievalEvaluationCase,
    evaluate_managed_retrieval,
    evaluation_cases_from_seed,
    write_retrieval_quality_report,
)


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
        help="Phase 2 seed evaluation JSON (defaults to evaluation.dataset_path).",
    )
    parser.add_argument("--k", type=_positive_int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def _build_current_runtime(config_path: Path) -> tuple[ManagedRetrievalRuntime, Path]:
    settings = load_settings(config_path)
    require_database_url(settings.database)
    engine = create_database_engine(settings.database)
    try:
        with Session(engine) as session:
            runtime = build_managed_retrieval_runtime(
                settings,
                LineageRepository(session),
                tracer=NoOpTracer(),
            )
    finally:
        engine.dispose()
    return runtime, settings.evaluation.dataset_path


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runtime, default_dataset_path = _build_current_runtime(args.config)
    dataset = load_evaluation_set(args.dataset or default_dataset_path)
    cases = evaluation_cases_from_seed(dataset)

    def show_progress(
        position: int,
        total: int,
        case: RetrievalEvaluationCase,
    ) -> None:
        print(f"Evaluating case {position}/{total}: {case.case_id}", flush=True)

    report = evaluate_managed_retrieval(
        runtime,
        cases,
        k=args.k,
        progress=show_progress,
    )

    print("Managed retrieval quality")
    print()
    print(f"Generation: g{report.generation:06d}")
    print(f"Collection: {report.qdrant_collection}")
    print(f"Cases: {report.case_count}")
    print()
    print(f"Recall@{report.metrics.k}: {report.recall_at_k:.4f}")
    print(f"MRR: {report.mrr:.4f}")
    print(f"Hit rate: {report.hit_rate:.4f}")
    print(f"Missed cases: {report.missed_case_count}")
    if args.output is not None:
        output = write_retrieval_quality_report(report, args.output)
        print(f"JSON report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
