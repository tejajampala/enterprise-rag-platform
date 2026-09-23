"""Inspect or explicitly publish the persisted Phase 3 managed corpus."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from uuid import UUID

from qdrant_client import QdrantClient
from sqlalchemy.exc import OperationalError, ProgrammingError
from sqlalchemy.orm import Session, sessionmaker

from consumer_finance_rag.config import AppSettings, load_settings
from consumer_finance_rag.lineage.chunk_materialization import (
    ActiveSnapshotResolutionError,
    ChunkMaterializationService,
)
from consumer_finance_rag.lineage.database import (
    DatabaseConfigurationError,
    create_database_engine,
    require_database_url,
)
from consumer_finance_rag.lineage.fingerprints import ManagedIndexConfiguration
from consumer_finance_rag.lineage.index_types import ActiveChunkManifest
from consumer_finance_rag.lineage.indexing import (
    ManagedIndexActivationConflictError,
    ManagedIndexBuildError,
    ManagedIndexPublishError,
    ManagedIndexPublishOutcome,
    ManagedIndexPublishResult,
    ManagedIndexService,
    ManagedIndexStaleLineageError,
)
from consumer_finance_rag.lineage.managed_vector_store import (
    ManagedCollectionCollisionError,
    ManagedQdrantError,
    ManagedQdrantStore,
)
from consumer_finance_rag.lineage.repository import (
    LineageNotFoundError,
    LineageRepository,
    LineageRepositoryError,
)
from consumer_finance_rag.providers import HuggingFaceEmbeddingProvider

SessionFactory = Callable[[], Session]


class Closable(Protocol):
    """Minimal lifecycle contract for the Qdrant client owned by this CLI."""

    def close(self, grpc_grace: float | None = None, **kwargs: object) -> None: ...


class ManagedIndexCLIError(RuntimeError):
    """Expected operator-state error with an actionable message."""


@dataclass(frozen=True, slots=True)
class ManagedIndexStatus:
    """Detached read-only status for deterministic terminal output."""

    corpus_key: str
    latest_run_id: UUID
    latest_run_status: str
    latest_snapshot_hash: str | None
    active_document_count: int
    active_index_state_id: UUID | None
    active_generation: int | None
    active_collection: str | None
    active_index_document_count: int | None
    active_index_chunk_count: int | None


@dataclass(frozen=True, slots=True)
class ManagedIndexRuntime:
    """Publisher plus the physical client whose lifecycle the CLI owns."""

    publisher: ManagedIndexService
    client: Closable


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/config.yaml"),
        help="Application YAML configuration path (default: config/config.yaml).",
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Build and activate an immutable managed Qdrant generation.",
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="Show additional persisted lineage/index identity (still read-only).",
    )
    return parser


def read_managed_index_status(
    session: Session,
    *,
    corpus_key: str,
) -> ManagedIndexStatus:
    """Read status only; never materialize chunks or contact Qdrant."""

    repository = LineageRepository(session)
    corpus = repository.get_corpus_by_key(corpus_key)
    if corpus is None:
        raise ManagedIndexCLIError(
            f"Managed corpus {corpus_key!r} is not initialized. Run "
            "`uv run python scripts/ingest_incremental.py --mode fixture --apply` "
            "(or the evaluation-mode apply command) first."
        )
    latest_run = repository.get_latest_successful_ingestion_run(corpus.corpus_id)
    if latest_run is None:
        raise ManagedIndexCLIError(
            f"Managed corpus {corpus_key!r} has no successful/no-op lineage run. "
            "Run `scripts/ingest_incremental.py ... --apply` successfully first."
        )
    active = repository.get_active_index_state(corpus.corpus_id)
    return ManagedIndexStatus(
        corpus_key=corpus.corpus_key,
        latest_run_id=latest_run.run_id,
        latest_run_status=latest_run.status,
        latest_snapshot_hash=latest_run.snapshot_hash,
        active_document_count=len(repository.list_active_documents(corpus.corpus_id)),
        active_index_state_id=(active.index_state_id if active is not None else None),
        active_generation=(active.generation if active is not None else None),
        active_collection=(active.qdrant_collection if active is not None else None),
        active_index_document_count=(
            active.document_count if active is not None else None
        ),
        active_index_chunk_count=(active.chunk_count if active is not None else None),
    )


def materialize_for_publication(
    session_factory: SessionFactory,
    settings: AppSettings,
) -> tuple[ActiveChunkManifest, UUID]:
    """Commit current chunk derivations before any publisher work starts."""

    with session_factory() as session, session.begin():
        repository = LineageRepository(session)
        corpus = repository.get_corpus_by_key(settings.managed_ingestion.corpus_key)
        if corpus is None:
            raise ManagedIndexCLIError(
                f"Managed corpus {settings.managed_ingestion.corpus_key!r} is not "
                "initialized; run incremental ingestion with --apply first."
            )
        latest_run = repository.get_latest_successful_ingestion_run(corpus.corpus_id)
        if latest_run is None:
            raise ManagedIndexCLIError(
                f"Managed corpus {corpus.corpus_key!r} has no successful/no-op "
                "lineage run; run incremental ingestion with --apply first."
            )
        target_run_id = latest_run.run_id
        manifest = ChunkMaterializationService(
            session,
            chunk_size=settings.chunking.chunk_size,
            overlap=settings.chunking.overlap,
        ).materialize_active_manifest(corpus.corpus_key)
    return manifest, target_run_id


def build_managed_index_runtime(
    settings: AppSettings,
    session_factory: SessionFactory,
) -> ManagedIndexRuntime:
    """Construct existing lazy model and managed-Qdrant adapters from settings."""

    embedding = HuggingFaceEmbeddingProvider(
        settings.models.embedding.model,
        settings.models.embedding.device,
        settings.models.embedding.trust_remote_code,
        settings.models.embedding.query_prompt_name,
    )
    client = QdrantClient(url=settings.qdrant.url)

    def store_factory(
        managed_prefix: str,
        configuration: ManagedIndexConfiguration,
        embedding_dimension: int,
    ) -> ManagedQdrantStore:
        return ManagedQdrantStore(
            client,
            managed_prefix=managed_prefix,
            configuration=configuration,
            embedding_dimension=embedding_dimension,
        )

    publisher = ManagedIndexService(
        session_factory=session_factory,
        embedding_provider=embedding,
        embedding_settings=settings.models.embedding,
        chunking_settings=settings.chunking,
        managed_index_settings=settings.managed_index,
        store_factory=store_factory,
    )
    return ManagedIndexRuntime(publisher=publisher, client=client)


def format_status(status: ManagedIndexStatus, *, details: bool) -> str:
    generation = (
        f"g{status.active_generation:06d}"
        if status.active_generation is not None
        else "none"
    )
    lines = [
        "Phase 3 managed index",
        "",
        f"Corpus key: {status.corpus_key}",
        f"Latest lineage run: {status.latest_run_id}",
        f"Latest lineage status: {status.latest_run_status}",
        f"Active logical documents: {status.active_document_count}",
        f"Active index generation: {generation}",
        f"Active collection: {status.active_collection or 'none'}",
    ]
    if details:
        lines.extend(
            [
                f"Latest snapshot hash: {status.latest_snapshot_hash or 'none'}",
                f"Active index state: {status.active_index_state_id or 'none'}",
                "Active index documents: "
                f"{_optional_count(status.active_index_document_count)}",
                f"Active index chunks: {_optional_count(status.active_index_chunk_count)}",
                "Physical Qdrant check: NOT REQUESTED",
            ]
        )
    lines.extend(
        [
            "",
            "Publishing: NOT REQUESTED",
            "",
            "Run with --publish to build/publish the desired managed generation.",
        ]
    )
    return "\n".join(lines)


def format_publish_result(result: ManagedIndexPublishResult, *, corpus_key: str) -> str:
    generation = f"g{result.generation:06d}"
    lines = [
        "Phase 3 managed index publication",
        "",
        f"Result: {result.outcome.value}",
        f"Corpus: {corpus_key}",
        f"Generation: {generation}",
        f"Collection: {result.collection}",
        f"Index state: {result.index_state_id}",
        "",
        f"Documents: {result.document_count}",
        f"Chunks: {result.chunk_count}",
    ]
    if result.outcome is ManagedIndexPublishOutcome.NO_INDEX_CHANGE:
        lines.extend(
            [
                "",
                "No Qdrant or index-state changes were required.",
                "Status: ACTIVE",
            ]
        )
    else:
        lines.extend(
            [
                "",
                f"BM25: {result.bm25_action.value}",
                f"Dense vectors reused: {result.dense_reused_count}",
                f"Dense vectors computed: {result.dense_computed_count}",
                "",
                "Status: ACTIVE",
                "Previous ACTIVE generation was retained.",
            ]
        )
    return "\n".join(lines)


def _optional_count(value: int | None) -> str:
    return str(value) if value is not None else "none"


def _publication_error(exc: ManagedIndexPublishError) -> str:
    if isinstance(exc, ManagedIndexStaleLineageError):
        return (
            "Managed lineage advanced during publication. Re-run the status command and "
            "publish the latest successful lineage state."
        )
    if isinstance(exc, ManagedIndexActivationConflictError):
        return (
            "Managed index activation conflicted with another generation. Inspect current "
            "status before retrying."
        )
    cause = exc.__cause__
    if isinstance(cause, ManagedCollectionCollisionError):
        return f"Managed Qdrant collection collision: {cause}"
    if isinstance(cause, ManagedQdrantError):
        return f"Managed Qdrant operation failed: {cause}"
    if isinstance(exc, ManagedIndexBuildError):
        detail = str(cause) if cause is not None else str(exc)
        return (
            "Managed index build failed during Qdrant or embedding/model work: "
            f"{detail}"
        )
    return f"Managed index publication failed: {exc}"


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = load_settings(Path(args.config))
    try:
        require_database_url(settings.database)
    except DatabaseConfigurationError as exc:
        print(f"Phase 3 managed-index configuration error: {exc}", file=sys.stderr)
        return 2

    engine = create_database_engine(settings.database)
    session_factory = sessionmaker(
        bind=engine,
        class_=Session,
        expire_on_commit=False,
    )
    runtime: ManagedIndexRuntime | None = None
    try:
        with session_factory() as session, session.begin():
            status = read_managed_index_status(
                session,
                corpus_key=settings.managed_ingestion.corpus_key,
            )
        if not args.publish:
            print(format_status(status, details=bool(args.details)))
            return 0

        print(
            "Phase 3 managed index publication\n\n"
            f"Corpus: {status.corpus_key}\n"
            f"Target lineage run: {status.latest_run_id}\n"
            "Materializing active chunks in PostgreSQL...",
            flush=True,
        )
        manifest, target_run_id = materialize_for_publication(session_factory, settings)
        runtime = build_managed_index_runtime(settings, session_factory)
        result = runtime.publisher.publish(manifest, target_run_id)
        print("", format_publish_result(result, corpus_key=status.corpus_key), sep="\n")
        return 0
    except (OperationalError, ProgrammingError):
        print(
            "Unable to read the Phase 3 application lineage tables. Verify the lineage "
            "PostgreSQL service and RAG__DATABASE__URL, then run "
            "`uv run alembic upgrade head`.",
            file=sys.stderr,
        )
        return 2
    except (ManagedIndexCLIError, ActiveSnapshotResolutionError, LineageNotFoundError) as exc:
        print(f"Managed index state error: {exc}", file=sys.stderr)
        return 2
    except ManagedIndexPublishError as exc:
        print(_publication_error(exc), file=sys.stderr)
        return 3
    except LineageRepositoryError as exc:
        print(f"Invalid Phase 3 lineage/index state: {exc}", file=sys.stderr)
        return 3
    finally:
        if runtime is not None:
            runtime.client.close()
        engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
