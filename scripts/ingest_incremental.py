"""Plan or persist Phase 3 managed-ingestion lineage without updating retrieval."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from sqlalchemy.exc import OperationalError, ProgrammingError
from sqlalchemy.orm import Session

from consumer_finance_rag.config import AppSettings, load_settings
from consumer_finance_rag.corpus import CorpusSnapshot, load_corpus_snapshot
from consumer_finance_rag.lineage.database import (
    DatabaseConfigurationError,
    create_database_engine,
    require_database_url,
)
from consumer_finance_rag.lineage.diff import PlannedDocumentChange
from consumer_finance_rag.lineage.repository import (
    LineageRepository,
    LineageRepositoryError,
)
from consumer_finance_rag.lineage.service import (
    LineageApplyResult,
    LineageApplyService,
    LineageDryRunResult,
    LineagePlanningService,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/config.yaml"),
        help="Application YAML configuration path (default: config/config.yaml).",
    )
    parser.add_argument(
        "--mode",
        choices=("fixture", "evaluation"),
        default="fixture",
        help="Plan the fixture corpus or fixture plus required downloaded evaluation corpus.",
    )
    parser.add_argument(
        "--authoritative-snapshot",
        action="store_true",
        help="Allow missing previously ACTIVE documents to be planned as DELETED.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Persist the displayed plan to PostgreSQL lineage tables only.",
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="Print every planned item without document text or chunks.",
    )
    return parser


def assemble_snapshot(settings: AppSettings, mode: str) -> CorpusSnapshot:
    """Reuse production corpus assembly with the same fixture/evaluation semantics."""

    evaluation_mode = mode == "evaluation"
    return load_corpus_snapshot(
        settings,
        include_expanded=evaluation_mode,
        require_expanded=evaluation_mode,
    )


def format_dry_run_summary(
    result: LineageDryRunResult,
    *,
    corpus_key: str,
    qdrant_collection_prefix: str,
    mode: str,
    details: bool,
    apply: bool = False,
) -> str:
    """Format the same deterministic plan for dry-run and apply modes."""

    plan = result.plan
    corpus_state = (
        "initialized"
        if result.corpus_initialized
        else "UNINITIALIZED (no PostgreSQL corpus row; previous state treated as empty)"
    )
    deletion_status = (
        "ENABLED (authoritative snapshot)"
        if plan.deletion_detection_performed
        else "SKIPPED (non-authoritative snapshot)"
    )
    lines = [
        f"Phase 3 managed ingestion {'APPLY' if apply else 'DRY RUN'}",
        "",
        f"Corpus key: {corpus_key}",
        f"Managed corpus state: {corpus_state}",
        f"Qdrant collection prefix: {qdrant_collection_prefix} (informational only)",
        f"Snapshot mode: {mode}",
        f"Snapshot hash: {result.snapshot_hash}",
        f"Documents in current snapshot: {result.current_document_count}",
        f"Previous active documents: {result.previous_active_document_count}",
        f"Authoritative snapshot: {'yes' if plan.authoritative_snapshot else 'no'}",
        f"Deletion detection: {deletion_status}",
        "",
        "Plan",
        "----",
        f"NEW:        {plan.new_count}",
        f"CHANGED:    {plan.changed_count}",
        f"UNCHANGED:  {plan.unchanged_count}",
        f"DELETED:    {plan.deleted_count}",
    ]
    if details:
        lines.extend(["", "Details", "-------"])
        for item in sorted(plan.items, key=lambda planned: planned.source_key):
            lines.extend(_format_detail(item))
    lines.extend(["", _plan_footer(apply)])
    return "\n".join(lines)


def _plan_footer(apply: bool) -> str:
    if apply:
        return "This plan will now be persisted to PostgreSQL lineage tables only."
    return "No PostgreSQL, Qdrant, BM25, or model changes were made."


def format_apply_result(result: LineageApplyResult) -> str:
    """Format a bounded account of lineage writes and explicitly omitted index work."""

    return "\n".join(
        [
            "Lineage apply completed.",
            "",
            f"Ingestion run: {result.run_id}",
            f"Run status: {result.status}",
            f"NEW persisted: {result.new_count}",
            f"CHANGED persisted: {result.changed_count}",
            f"UNCHANGED observed: {result.unchanged_count}",
            f"DELETED marked: {result.deleted_count}",
            "",
            "Qdrant indexing: NOT RUN",
            "BM25 update: NOT RUN",
            "Embedding inference: NOT RUN",
        ]
    )


def _format_detail(item: PlannedDocumentChange) -> list[str]:
    return [
        "",
        item.change_type.value,
        f"  source_key: {item.source_key}",
        f"  document_pk: {item.document_pk or '-'}",
        f"  previous_version: {item.previous_version_number or '-'}",
        f"  content_changed: {str(item.content_changed).lower()}",
        f"  metadata_changed: {str(item.metadata_changed).lower()}",
        f"  previous_content_hash: {item.previous_content_hash or '-'}",
        f"  current_content_hash: {item.current_content_hash or '-'}",
        f"  previous_metadata_hash: {item.previous_metadata_hash or '-'}",
        f"  current_metadata_hash: {item.current_metadata_hash or '-'}",
    ]


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = load_settings(Path(args.config))
    try:
        require_database_url(settings.database)
    except DatabaseConfigurationError as exc:
        print(f"Phase 3 ingestion configuration error: {exc}", file=sys.stderr)
        return 2

    try:
        snapshot = assemble_snapshot(settings, str(args.mode))
    except FileNotFoundError as exc:
        print(f"Phase 3 ingestion corpus error: {exc}", file=sys.stderr)
        return 2

    engine = create_database_engine(settings.database)
    try:
        with Session(engine) as session:
            with session.begin():
                result = LineagePlanningService(LineageRepository(session)).plan_snapshot(
                    snapshot,
                    corpus_key=settings.managed_ingestion.corpus_key,
                    authoritative_snapshot=bool(args.authoritative_snapshot),
                )
            print(
                format_dry_run_summary(
                    result,
                    corpus_key=settings.managed_ingestion.corpus_key,
                    qdrant_collection_prefix=(
                        settings.managed_ingestion.qdrant_collection_prefix
                    ),
                    mode=str(args.mode),
                    details=bool(args.details),
                    apply=bool(args.apply),
                ),
                flush=True,
            )
            apply_result = None
            if args.apply:
                apply_result = LineageApplyService(
                    session,
                    chunk_size=settings.chunking.chunk_size,
                    overlap=settings.chunking.overlap,
                ).apply_plan(
                    result.plan,
                    corpus_key=settings.managed_ingestion.corpus_key,
                    qdrant_collection_prefix=(
                        settings.managed_ingestion.qdrant_collection_prefix
                    ),
                    snapshot_hash=result.snapshot_hash,
                )
    except (OperationalError, ProgrammingError):
        print(
            "Unable to read the Phase 3 application lineage tables. Verify the database "
            "service and RAG__DATABASE__URL, then run `uv run alembic upgrade head`.",
            file=sys.stderr,
        )
        return 2
    except LineageRepositoryError as exc:
        print(f"Invalid Phase 3 lineage state: {exc}", file=sys.stderr)
        return 2
    finally:
        engine.dispose()

    if apply_result is not None:
        print("", format_apply_result(apply_result), sep="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
