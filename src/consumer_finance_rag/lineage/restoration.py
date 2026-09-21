"""Safe restoration of historical content into a new immutable generation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from typing import Any, cast
from uuid import UUID

from qdrant_client import QdrantClient
from sqlalchemy.orm import Session

from consumer_finance_rag.config import EmbeddingModelSettings, ManagedIndexSettings
from consumer_finance_rag.lineage.fingerprints import (
    ManagedIndexConfiguration,
    embedding_fingerprint,
)
from consumer_finance_rag.lineage.index_operations import (
    ManagedIndexOperationsService,
)
from consumer_finance_rag.lineage.index_types import (
    DensePlanAction,
    PreparedIndexMembership,
)
from consumer_finance_rag.lineage.managed_retrieval import (
    CorruptActiveManagedIndexError,
    validate_active_managed_index_metadata,
)
from consumer_finance_rag.lineage.managed_vector_store import (
    ManagedQdrantStore,
    managed_generation_collection_name,
)
from consumer_finance_rag.lineage.models import IndexState, utc_now
from consumer_finance_rag.lineage.repository import LineageRepository

SessionFactory = Callable[[], Session]
Clock = Callable[[], datetime]


class ManagedGenerationRestorationError(RuntimeError):
    """Base exception for safe managed-generation restoration."""


class RestorationSourceIneligibleError(ManagedGenerationRestorationError):
    """Raised when the requested source is not a RETIRED generation."""


class RestorationUnavailablePhysicalCleanedError(ManagedGenerationRestorationError):
    """Raised when PostgreSQL history remains but its collection was cleaned."""


class RestorationValidationError(ManagedGenerationRestorationError):
    """Raised when source, active, configuration, or physical data is invalid."""


class RestorationActivationConflictError(ManagedGenerationRestorationError):
    """Raised when ACTIVE or generation allocation changes during restoration."""


class RestorationBuildError(ManagedGenerationRestorationError):
    """Raised after a restoration candidate fails before READY."""


@dataclass(frozen=True, slots=True)
class GenerationRestorationPlan:
    """Immutable, mutation-free proposal for one historical generation restore."""

    corpus_key: str
    source_generation: int
    source_index_state_id: UUID
    source_collection: str
    current_active_generation: int
    current_active_index_state_id: UUID
    proposed_generation: int
    proposed_target_collection: str
    document_count: int
    chunk_count: int
    bm25_state_hash: str
    embedding_fingerprint: str
    configuration_fingerprint: str
    payload_schema_version: str
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GenerationRestorationResult:
    """Successful restoration and traffic-switch result."""

    source_generation: int
    source_index_state_id: UUID
    previous_active_generation: int
    previous_active_index_state_id: UUID
    new_active_generation: int
    new_active_index_state_id: UUID
    collection: str
    dense_vectors_copied: int
    chunk_count: int


@dataclass(frozen=True, slots=True)
class _SourceSnapshot:
    index_state_id: UUID
    ingestion_run_id: UUID
    snapshot_hash: str
    embedding_fingerprint: str
    embedding_dimension: int
    chunking_fingerprint: str
    document_count: int
    chunk_count: int
    bm25_state: Mapping[str, Any]
    bm25_state_hash: str
    bm25_corpus_hash: str
    index_configuration: Mapping[str, Any]
    index_configuration_fingerprint: str
    configuration: ManagedIndexConfiguration
    memberships: tuple[PreparedIndexMembership, ...]


class ManagedGenerationRestorationService:
    """Copy a valid RETIRED generation and atomically activate the new copy."""

    def __init__(
        self,
        *,
        session_factory: SessionFactory,
        qdrant_client: QdrantClient,
        corpus_key: str,
        managed_index_settings: ManagedIndexSettings,
        embedding_settings: EmbeddingModelSettings,
        clock: Clock = utc_now,
    ) -> None:
        self._session_factory = session_factory
        self._client = qdrant_client
        self._corpus_key = corpus_key
        self._managed_index_settings = managed_index_settings
        self._embedding_settings = embedding_settings
        self._clock = clock
        self._operations = ManagedIndexOperationsService(
            session_factory=session_factory,
            qdrant_client=qdrant_client,
            corpus_key=corpus_key,
            settings=managed_index_settings,
            clock=clock,
        )

    def plan(self, source_generation: int) -> GenerationRestorationPlan:
        """Validate both planes and return a plan without making mutations."""

        if source_generation <= 0:
            raise RestorationSourceIneligibleError(
                "Restore source generation must be positive"
            )
        with self._session_factory() as session:
            repository = LineageRepository(session)
            corpus = repository.get_corpus_by_key(self._corpus_key)
            if corpus is None:
                raise ManagedGenerationRestorationError(
                    f"Managed corpus {self._corpus_key!r} is not initialized"
                )
            states = repository.list_index_states(corpus.corpus_id)
            source = next(
                (state for state in states if state.generation == source_generation),
                None,
            )
            if source is None:
                raise RestorationSourceIneligibleError(
                    f"Managed generation g{source_generation:06d} does not exist"
                )
            if source.status != "RETIRED":
                if source.status == "ACTIVE":
                    detail = "ACTIVE generation is already serving and cannot be restored"
                else:
                    detail = f"{source.status} generations cannot be restoration sources"
                raise RestorationSourceIneligibleError(detail)
            active = repository.get_active_index_state(corpus.corpus_id)
            if active is None:
                raise RestorationValidationError(
                    "Restoration requires one current ACTIVE generation"
                )
            proposed_generation = repository.get_latest_generation_number(corpus.corpus_id) + 1
            target_collection = managed_generation_collection_name(
                corpus.qdrant_collection_prefix,
                proposed_generation,
            )
            expected_source_collection = managed_generation_collection_name(
                corpus.qdrant_collection_prefix,
                source_generation,
            )
            source_id = source.index_state_id
            source_collection = source.qdrant_collection
            active_id = active.index_state_id
            active_generation = active.generation
            latest_run = repository.get_latest_successful_ingestion_run(
                corpus.corpus_id
            )
            latest_snapshot_hash = (
                latest_run.snapshot_hash if latest_run is not None else None
            )

        if source_collection != expected_source_collection:
            raise RestorationValidationError(
                "Historical source collection name is not canonical for its generation"
            )

        source_validation = self._operations.validate_generation(
            source_generation,
            deep=True,
        )
        if not source_validation.physical_present:
            raise RestorationUnavailablePhysicalCleanedError(
                "Historical generation exists in PostgreSQL but its physical collection "
                "has been cleaned and cannot be restored by this operation. "
                "RESTORATION_UNAVAILABLE_PHYSICAL_CLEANED"
            )
        if not source_validation.valid:
            raise RestorationValidationError(
                "Historical source generation failed validation: "
                + "; ".join(source_validation.issues)
            )
        active_validation = self._operations.validate_generation(
            active_generation,
            deep=False,
        )
        if not active_validation.valid:
            raise RestorationValidationError(
                "Current ACTIVE generation failed validation: "
                + "; ".join(active_validation.issues)
            )
        try:
            target_exists = bool(self._client.collection_exists(target_collection))
        except Exception as exc:
            raise RestorationValidationError(
                f"Unable to inspect proposed target collection: {exc}"
            ) from exc
        if target_exists:
            raise RestorationValidationError(
                f"Proposed target collection already exists: {target_collection}"
            )

        with self._session_factory() as session:
            source = LineageRepository(session).get_index_state(source_id)
            if source is None:
                raise RestorationActivationConflictError(
                    "Historical source generation disappeared during planning"
                )
            try:
                configuration, _ = validate_active_managed_index_metadata(source)
            except CorruptActiveManagedIndexError as exc:
                raise RestorationValidationError(
                    f"Historical source metadata is incompatible: {exc}"
                ) from exc
            self._require_runtime_embedding_compatibility(source, configuration)
            bm25_hash = cast(str, source.bm25_state_hash)
            configuration_hash = cast(
                str,
                source.index_configuration_fingerprint,
            )
            document_count = source.document_count
            chunk_count = source.chunk_count
            embedding_identity = source.embedding_fingerprint

        warnings = [
            "Source lineage is unchanged; restoration only changes the serving index.",
            "Deleting the historical physical collection removes simple restoration capability.",
        ]
        if latest_snapshot_hash != source.snapshot_hash:
            warnings.append(
                "Current source lineage/index content may be newer than the historical content."
            )
        return GenerationRestorationPlan(
            corpus_key=self._corpus_key,
            source_generation=source_generation,
            source_index_state_id=source_id,
            source_collection=source_collection,
            current_active_generation=active_generation,
            current_active_index_state_id=active_id,
            proposed_generation=proposed_generation,
            proposed_target_collection=target_collection,
            document_count=document_count,
            chunk_count=chunk_count,
            bm25_state_hash=bm25_hash,
            embedding_fingerprint=embedding_identity,
            configuration_fingerprint=configuration_hash,
            payload_schema_version=configuration.payload_schema_version,
            warnings=tuple(warnings),
        )

    def execute(
        self,
        plan: GenerationRestorationPlan,
        *,
        restoration_reason: str,
    ) -> GenerationRestorationResult:
        """Create, validate, and atomically activate a new restored generation."""

        if plan.corpus_key != self._corpus_key:
            raise RestorationActivationConflictError(
                "Restoration plan corpus does not match this service"
            )
        if not restoration_reason.strip():
            raise ManagedGenerationRestorationError(
                "Confirmed restoration requires a non-empty restoration reason"
            )
        current_plan = self.plan(plan.source_generation)
        if current_plan != plan:
            raise RestorationActivationConflictError(
                "Restoration plan is stale; ACTIVE generation or allocation changed"
            )
        candidate_id = self._allocate_candidate(plan, restoration_reason.strip())
        source = self._load_source_snapshot(plan.source_index_state_id)
        store = ManagedQdrantStore(
            self._client,
            managed_prefix=self._managed_prefix(),
            configuration=source.configuration,
            embedding_dimension=source.embedding_dimension,
        )
        copied = 0
        try:
            store.create_generation_collection(plan.proposed_target_collection)
            copied = store.copy_generation_points(
                plan.source_collection,
                plan.proposed_target_collection,
                target_index_state_id=candidate_id,
            )
            if copied != source.chunk_count:
                raise RestorationValidationError(
                    f"Copied point count {copied} != {source.chunk_count}"
                )
            store.validate_restored_generation_collection(
                plan.source_collection,
                plan.proposed_target_collection,
                target_index_state_id=candidate_id,
            )
            self._persist_ready(candidate_id, source)
        except Exception as exc:
            self._mark_failed_preserving_original(candidate_id, "BUILDING", exc)
            raise RestorationBuildError(
                f"Restored generation g{plan.proposed_generation:06d} failed before READY"
            ) from exc

        try:
            self._activate(candidate_id, plan)
        except Exception as exc:
            self._mark_failed_preserving_original(candidate_id, "READY", exc)
            if isinstance(exc, RestorationActivationConflictError):
                raise
            raise RestorationActivationConflictError(
                "Restored candidate could not be atomically activated"
            ) from exc
        return GenerationRestorationResult(
            source_generation=plan.source_generation,
            source_index_state_id=plan.source_index_state_id,
            previous_active_generation=plan.current_active_generation,
            previous_active_index_state_id=plan.current_active_index_state_id,
            new_active_generation=plan.proposed_generation,
            new_active_index_state_id=candidate_id,
            collection=plan.proposed_target_collection,
            dense_vectors_copied=copied,
            chunk_count=source.chunk_count,
        )

    def _allocate_candidate(
        self,
        plan: GenerationRestorationPlan,
        restoration_reason: str,
    ) -> UUID:
        with self._session_factory() as session, session.begin():
            repository = LineageRepository(session)
            corpus = repository.get_corpus_by_key(self._corpus_key)
            if corpus is None:
                raise RestorationActivationConflictError("Managed corpus disappeared")
            repository.lock_corpus(corpus.corpus_id)
            source = repository.get_index_state(plan.source_index_state_id)
            if source is None or source.status != "RETIRED":
                raise RestorationActivationConflictError(
                    "Restoration source is no longer RETIRED"
                )
            active = repository.get_active_index_state(corpus.corpus_id)
            if active is None or active.index_state_id != plan.current_active_index_state_id:
                raise RestorationActivationConflictError(
                    "ACTIVE generation changed before restoration allocation"
                )
            generation = repository.get_latest_generation_number(corpus.corpus_id) + 1
            collection = managed_generation_collection_name(
                corpus.qdrant_collection_prefix,
                generation,
            )
            if (
                generation != plan.proposed_generation
                or collection != plan.proposed_target_collection
            ):
                raise RestorationActivationConflictError(
                    "Next monotonic generation changed before restoration allocation"
                )
            candidate = repository.create_building_index_state(
                corpus.corpus_id,
                generation,
                source.ingestion_run_id,
                collection,
                source.snapshot_hash,
                source.embedding_fingerprint,
                source.chunking_fingerprint,
                document_count=source.document_count,
                chunk_count=source.chunk_count,
                restored_from_index_state_id=source.index_state_id,
                restoration_reason=restoration_reason,
            )
            return candidate.index_state_id

    def _load_source_snapshot(self, source_id: UUID) -> _SourceSnapshot:
        with self._session_factory() as session:
            repository = LineageRepository(session)
            source = repository.get_index_state(source_id)
            if source is None or source.status != "RETIRED":
                raise RestorationActivationConflictError(
                    "Restoration source is no longer RETIRED"
                )
            try:
                configuration, _ = validate_active_managed_index_metadata(source)
            except CorruptActiveManagedIndexError as exc:
                raise RestorationValidationError(str(exc)) from exc
            if source.embedding_dimension is None:
                raise RestorationValidationError("Source embedding dimension is missing")
            memberships = tuple(
                PreparedIndexMembership(
                    chunk_pk=item.chunk_pk,
                    qdrant_point_id=item.qdrant_point_id,
                    dense_action=DensePlanAction.REUSED,
                    dense_source_chunk_pk=item.chunk_pk,
                )
                for item in repository.list_index_state_chunks(source_id)
            )
            return _SourceSnapshot(
                index_state_id=source.index_state_id,
                ingestion_run_id=source.ingestion_run_id,
                snapshot_hash=source.snapshot_hash,
                embedding_fingerprint=source.embedding_fingerprint,
                embedding_dimension=source.embedding_dimension,
                chunking_fingerprint=source.chunking_fingerprint,
                document_count=source.document_count,
                chunk_count=source.chunk_count,
                bm25_state=deepcopy(cast(Mapping[str, Any], source.bm25_state)),
                bm25_state_hash=cast(str, source.bm25_state_hash),
                bm25_corpus_hash=cast(str, source.bm25_corpus_hash),
                index_configuration=deepcopy(
                    cast(Mapping[str, Any], source.index_configuration)
                ),
                index_configuration_fingerprint=cast(
                    str,
                    source.index_configuration_fingerprint,
                ),
                configuration=configuration,
                memberships=memberships,
            )

    def _persist_ready(self, candidate_id: UUID, source: _SourceSnapshot) -> None:
        with self._session_factory() as session, session.begin():
            repository = LineageRepository(session)
            candidate = repository.get_index_state(candidate_id)
            historical = repository.get_index_state(source.index_state_id)
            if candidate is None or candidate.status != "BUILDING":
                raise RestorationActivationConflictError(
                    "Restoration candidate is no longer BUILDING"
                )
            if historical is None or historical.status != "RETIRED":
                raise RestorationActivationConflictError(
                    "Restoration source is no longer RETIRED"
                )
            repository.update_index_build_metadata(
                candidate_id,
                bm25_state=source.bm25_state,
                bm25_state_hash=source.bm25_state_hash,
                bm25_corpus_hash=source.bm25_corpus_hash,
                index_configuration=source.index_configuration,
                index_configuration_fingerprint=(
                    source.index_configuration_fingerprint
                ),
                embedding_dimension=source.embedding_dimension,
                document_count=source.document_count,
                chunk_count=source.chunk_count,
            )
            repository.record_index_state_chunks_batch(
                candidate_id,
                source.memberships,
            )
            repository.transition_index_state(candidate_id, "BUILDING", "READY")

    def _activate(
        self,
        candidate_id: UUID,
        plan: GenerationRestorationPlan,
    ) -> None:
        with self._session_factory() as session, session.begin():
            repository = LineageRepository(session)
            corpus = repository.get_corpus_by_key(self._corpus_key)
            if corpus is None:
                raise RestorationActivationConflictError("Managed corpus disappeared")
            repository.lock_corpus(corpus.corpus_id)
            candidate = repository.get_index_state(candidate_id)
            if candidate is None or candidate.status != "READY":
                raise RestorationActivationConflictError(
                    "Restoration candidate is no longer READY"
                )
            if (
                candidate.generation != plan.proposed_generation
                or candidate.qdrant_collection != plan.proposed_target_collection
                or candidate.restored_from_index_state_id
                != plan.source_index_state_id
                or candidate.index_configuration_fingerprint
                != plan.configuration_fingerprint
            ):
                raise RestorationActivationConflictError(
                    "Restoration candidate identity changed before activation"
                )
            source = repository.get_index_state(plan.source_index_state_id)
            if source is None or source.status != "RETIRED":
                raise RestorationActivationConflictError(
                    "Historical source is no longer RETIRED"
                )
            current = repository.get_active_index_state(corpus.corpus_id)
            if current is None or current.index_state_id != plan.current_active_index_state_id:
                raise RestorationActivationConflictError(
                    "A different ACTIVE generation replaced the restoration baseline"
                )
            occurred_at = self._clock()
            repository.transition_index_state(
                current.index_state_id,
                "ACTIVE",
                "RETIRED",
                occurred_at=occurred_at,
            )
            repository.transition_index_state(
                candidate_id,
                "READY",
                "ACTIVE",
                occurred_at=occurred_at,
            )

    def _mark_failed_preserving_original(
        self,
        candidate_id: UUID,
        expected_status: str,
        original: Exception,
    ) -> None:
        try:
            with self._session_factory() as session, session.begin():
                LineageRepository(session).mark_index_state_failed(
                    candidate_id,
                    expected_status,
                    str(original) or type(original).__name__,
                    self._clock(),
                )
        except Exception as failure_recording_error:
            original.add_note(
                "Additionally failed to persist restoration candidate FAILED state: "
                f"{failure_recording_error}"
            )

    def _managed_prefix(self) -> str:
        with self._session_factory() as session:
            corpus = LineageRepository(session).get_corpus_by_key(self._corpus_key)
            if corpus is None:
                raise ManagedGenerationRestorationError("Managed corpus disappeared")
            return corpus.qdrant_collection_prefix

    def _require_runtime_embedding_compatibility(
        self,
        source: IndexState,
        configuration: ManagedIndexConfiguration,
    ) -> None:
        if (
            configuration.managed_index_format_version
            != self._managed_index_settings.format_version
        ):
            raise RestorationValidationError(
                "Historical managed index format is incompatible with the current runtime"
            )
        runtime_fingerprint = embedding_fingerprint(
            provider=self._embedding_settings.provider,
            model=self._embedding_settings.model,
            trust_remote_code=self._embedding_settings.trust_remote_code,
            device=self._embedding_settings.device,
        )
        if (
            runtime_fingerprint != source.embedding_fingerprint
            or runtime_fingerprint != configuration.embedding_fingerprint
        ):
            raise RestorationValidationError(
                "Configured managed runtime embedding fingerprint cannot query the "
                "historical generation"
            )
