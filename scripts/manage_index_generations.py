"""Inspect, validate, or explicitly clean managed Qdrant generations."""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from qdrant_client import QdrantClient
from sqlalchemy.exc import OperationalError, ProgrammingError
from sqlalchemy.orm import Session, sessionmaker

from consumer_finance_rag.config import load_settings
from consumer_finance_rag.lineage.database import (
    DatabaseConfigurationError,
    create_database_engine,
    require_database_url,
)
from consumer_finance_rag.lineage.index_operations import (
    GenerationCleanupExecutionResult,
    GenerationCleanupPlan,
    ManagedGenerationStatus,
    ManagedGenerationValidationReport,
    ManagedIndexOperationsError,
    ManagedIndexOperationsService,
)
from consumer_finance_rag.lineage.restoration import (
    GenerationRestorationPlan,
    GenerationRestorationResult,
    ManagedGenerationRestorationError,
    ManagedGenerationRestorationService,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/config.yaml"),
        help="Application YAML configuration path (default: config/config.yaml).",
    )
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--status", action="store_true", help="Show all generations.")
    actions.add_argument(
        "--validate",
        action="store_true",
        help="Validate ACTIVE and bounded historical generation invariants.",
    )
    actions.add_argument(
        "--cleanup-plan",
        action="store_true",
        help="Print a read-only physical cleanup plan.",
    )
    actions.add_argument(
        "--cleanup",
        action="store_true",
        help="Plan cleanup; deletion still requires --confirm.",
    )
    actions.add_argument(
        "--restore-generation",
        type=int,
        metavar="GENERATION",
        help="Plan restoration of one RETIRED generation into a new generation.",
    )
    parser.add_argument(
        "--deep",
        action="store_true",
        help="With --validate, scroll and validate every generation point.",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="With --cleanup or --restore-generation, permit the planned change.",
    )
    return parser


def format_status(statuses: Sequence[ManagedGenerationStatus]) -> str:
    lines = ["Managed generations", ""]
    for status in statuses:
        physical = (
            "present"
            if status.physical_present
            else "cleaned"
            if status.status == "RETIRED"
            else "missing"
        )
        lines.extend(
            [
                f"g{status.generation:06d} {status.status}",
                f"  collection: {status.qdrant_collection}",
                f"  physical: {physical}",
                f"  cleanup: {status.cleanup_classification.value.lower()}",
                f"  ingestion run: {status.ingestion_run_id}",
                f"  documents: {status.document_count}",
                f"  chunks: {status.chunk_count}",
                f"  embedding dimension: {_optional(status.embedding_dimension)}",
                f"  created: {_timestamp(status.created_at)}",
                f"  activated: {_timestamp(status.activated_at)}",
                f"  retired: {_timestamp(status.retired_at)}",
                f"  failed: {_timestamp(status.failed_at)}",
            ]
        )
        if status.error_message:
            lines.append(f"  error: {_safe_error(status.error_message)}")
        if status.restored_from_index_state_id is not None:
            lines.append(f"  restored from: {status.restored_from_index_state_id}")
            lines.append(
                f"  restoration reason: {_safe_error(status.restoration_reason or '')}"
            )
        lines.append("")
    active = sum(status.status == "ACTIVE" for status in statuses)
    eligible = sum(
        status.cleanup_classification.value.startswith("ELIGIBLE")
        for status in statuses
    )
    failed = sum(status.status == "FAILED" for status in statuses)
    lines.extend(
        [
            "Summary:",
            f"{active} ACTIVE",
            f"{eligible} eligible physical collections",
            f"{failed} failed historical generations",
        ]
    )
    return "\n".join(lines)


def format_validation(report: ManagedGenerationValidationReport) -> str:
    mode = "deep" if any(item.deep for item in report.generations) else "bounded"
    lines = [f"Managed generation validation ({mode})", ""]
    for issue in report.issues:
        lines.append(f"ERROR: {issue}")
    for result in report.generations:
        outcome = result.outcome
        physical = "present" if result.physical_present else "missing"
        lines.append(
            f"g{result.generation:06d} {result.status} {outcome} physical={physical} "
            f"classification={result.classification.value}"
        )
        lines.extend(f"  issue: {issue}" for issue in result.issues)
        lines.extend(f"  note: {note}" for note in result.notes)
    lines.extend(
        [
            "",
            f"ACTIVE validation: {report.active_health.value}",
            f"Historical audit: {report.historical_audit}",
            f"Overall: {report.overall_status.value}",
        ]
    )
    return "\n".join(lines)


def format_cleanup_plan(plan: GenerationCleanupPlan) -> str:
    lines = [
        "Managed generation cleanup plan (READ ONLY)",
        "",
        f"Corpus: {plan.corpus_key}",
        f"Retained RETIRED generations: {plan.retained_retired_generations}",
        f"Minimum RETIRED age hours: {plan.retired_cleanup_min_age_hours:g}",
        "",
    ]
    for candidate in plan.candidates:
        physical = "present" if candidate.physical_present else "missing"
        lines.append(
            f"g{candidate.generation:06d} {candidate.status} physical={physical} "
            f"cleanup={candidate.classification.value}"
        )
        lines.append(f"  collection: {candidate.qdrant_collection}")
        lines.append(f"  reason: {candidate.reason}")
    lines.extend(
        [
            "",
            f"Eligible physical collections: {len(plan.eligible)}",
            f"Detected orphans (report-only): {len(plan.orphans)}",
            "PostgreSQL history deletion: NEVER",
        ]
    )
    return "\n".join(lines)


def format_cleanup_result(result: GenerationCleanupExecutionResult) -> str:
    heading = "CONFIRMED CLEANUP" if result.confirmed else "CLEANUP NOT CONFIRMED"
    lines = [heading, ""]
    for item in result.items:
        outcome = "DELETED" if item.deleted else "SKIPPED"
        lines.append(
            f"g{item.generation:06d} {outcome} {item.qdrant_collection}: {item.reason}"
        )
    lines.extend(
        [
            "",
            f"Physical collections deleted: {len(result.deleted_collections)}",
            "PostgreSQL lineage/index history: PRESERVED",
        ]
    )
    return "\n".join(lines)


def format_restoration_plan(plan: GenerationRestorationPlan) -> str:
    lines = [
        "Restore plan",
        "",
        "Source generation:",
        f"g{plan.source_generation:06d} RETIRED",
        plan.source_collection,
        "",
        "Current ACTIVE:",
        f"g{plan.current_active_generation:06d}",
        "",
        "Proposed generation:",
        f"g{plan.proposed_generation:06d}",
        "",
        "Target collection:",
        plan.proposed_target_collection,
        "",
        f"Documents: {plan.document_count}",
        f"Chunks: {plan.chunk_count}",
        "Embedding: compatible",
        "BM25: valid / copied",
        f"ACL/PII schema: compatible ({plan.payload_schema_version})",
    ]
    lines.extend(f"Warning: {warning}" for warning in plan.warnings)
    lines.extend(["", "Action: NO CHANGES YET — rerun with --confirm"])
    return "\n".join(lines)


def format_restoration_result(result: GenerationRestorationResult) -> str:
    return "\n".join(
        [
            "Restore completed",
            "",
            "Source:",
            f"g{result.source_generation:06d} (remains RETIRED)",
            "",
            "Previous ACTIVE:",
            f"g{result.previous_active_generation:06d} -> RETIRED",
            "",
            "New ACTIVE:",
            f"g{result.new_active_generation:06d}",
            "",
            "Collection:",
            result.collection,
            "",
            f"Dense vectors: copied/reused {result.dense_vectors_copied}",
            "BM25: copied",
            "Source lineage: unchanged",
        ]
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.deep and not args.validate:
        print("--deep requires --validate", file=sys.stderr)
        return 2
    if args.confirm and not (args.cleanup or args.restore_generation is not None):
        print(
            "--confirm is only valid with --cleanup or --restore-generation",
            file=sys.stderr,
        )
        return 2

    settings = load_settings(args.config)
    try:
        require_database_url(settings.database)
    except DatabaseConfigurationError as exc:
        print(f"Managed operations configuration error: {exc}", file=sys.stderr)
        return 2

    engine = create_database_engine(settings.database)
    sessions = sessionmaker(bind=engine, class_=Session, expire_on_commit=False)
    client = QdrantClient(url=settings.qdrant.url)
    service = ManagedIndexOperationsService(
        session_factory=sessions,
        qdrant_client=client,
        corpus_key=settings.managed_ingestion.corpus_key,
        settings=settings.managed_index,
    )
    try:
        if args.validate:
            report = service.validate(deep=bool(args.deep))
            print(format_validation(report))
            return 0 if report.valid else 3
        if args.cleanup_plan:
            print(format_cleanup_plan(service.cleanup_plan()))
            return 0
        if args.cleanup:
            cleanup_plan = service.cleanup_plan()
            print(format_cleanup_plan(cleanup_plan))
            cleanup_result = service.execute_cleanup(
                cleanup_plan,
                confirmed=bool(args.confirm),
            )
            print("", format_cleanup_result(cleanup_result), sep="\n")
            return (
                0
                if all(item.deleted for item in cleanup_result.items) or not args.confirm
                else 3
            )
        if args.restore_generation is not None:
            restoration = ManagedGenerationRestorationService(
                session_factory=sessions,
                qdrant_client=client,
                corpus_key=settings.managed_ingestion.corpus_key,
                managed_index_settings=settings.managed_index,
                embedding_settings=settings.models.embedding,
            )
            restoration_plan = restoration.plan(args.restore_generation)
            print(format_restoration_plan(restoration_plan))
            if not args.confirm:
                return 0
            restoration_result = restoration.execute(
                restoration_plan,
                restoration_reason=(
                    f"Operator-confirmed restoration of g{args.restore_generation:06d}"
                ),
            )
            print("", format_restoration_result(restoration_result), sep="\n")
            return 0
        print(format_status(service.status()))
        return 0
    except (OperationalError, ProgrammingError):
        print(
            "Unable to read Phase 3 lineage tables. Verify PostgreSQL, "
            "RAG__DATABASE__URL, and `uv run alembic upgrade head`.",
            file=sys.stderr,
        )
        return 2
    except (ManagedIndexOperationsError, ManagedGenerationRestorationError) as exc:
        print(f"Managed generation operation failed: {exc}", file=sys.stderr)
        return 3
    finally:
        client.close()
        engine.dispose()


def _optional(value: object | None) -> str:
    return str(value) if value is not None else "none"


def _timestamp(value: datetime | None) -> str:
    return value.isoformat() if value is not None else "none"


def _safe_error(value: str) -> str:
    redacted = re.sub(
        r"(?i)(postgresql(?:\+[a-z0-9]+)?://)[^@\s]+@",
        r"\1***@",
        value,
    )
    redacted = re.sub(
        r"(?i)\b(password|api[_-]?key|secret)=\S+",
        r"\1=***",
        redacted,
    )
    return redacted[:500]


if __name__ == "__main__":
    raise SystemExit(main())
