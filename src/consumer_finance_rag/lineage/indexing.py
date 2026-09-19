"""Transactional publisher for immutable managed hybrid-index generations."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime
from enum import StrEnum
from typing import Any, Protocol
from uuid import UUID

from sqlalchemy.orm import Session

from consumer_finance_rag.config import (
    ChunkingSettings,
    EmbeddingModelSettings,
    ManagedIndexSettings,
)
from consumer_finance_rag.lineage.bm25_generation import plan_bm25_generation
from consumer_finance_rag.lineage.dense_generation import (
    DenseGenerationPlanner,
    DenseVectorReader,
)
from consumer_finance_rag.lineage.fingerprints import (
    ManagedIndexConfiguration,
    ManagedIndexConfigurationIdentity,
    chunking_fingerprint,
    embedding_fingerprint,
    managed_index_configuration_identity_from_settings,
)
from consumer_finance_rag.lineage.index_types import (
    ActiveChunkManifest,
    BM25PlanAction,
    DenseReuseCandidate,
    DenseVectorRetrievalResult,
    ManagedCollectionMetadata,
    ManagedCollectionValidationResult,
    ManagedPointInput,
    PreparedIndexIdentity,
    PreparedIndexMembership,
    PreviousBM25GenerationState,
    PreviousDenseGenerationState,
)
from consumer_finance_rag.lineage.managed_vector_store import (
    managed_generation_collection_name,
)
from consumer_finance_rag.lineage.models import IndexState, utc_now
from consumer_finance_rag.lineage.point_preparation import (
    prepare_managed_generation,
)
from consumer_finance_rag.lineage.repository import (
    FINISHED_INGESTION_STATUSES,
    LineageNotFoundError,
    LineageRepository,
)
from consumer_finance_rag.providers import EmbeddingProvider
from consumer_finance_rag.sparse import BM25State, bm25_state_to_json_dict

SessionFactory = Callable[[], Session]
Clock = Callable[[], datetime]


class ManagedIndexPublishError(RuntimeError):
    """Base exception for managed-index publication failures."""


class ManagedIndexBuildError(ManagedIndexPublishError):
    """Raised after a candidate fails before reaching READY."""


class ManagedIndexActivationConflictError(ManagedIndexPublishError):
    """Raised when the build baseline no longer owns activation."""


class ManagedIndexStaleLineageError(ManagedIndexActivationConflictError):
    """Raised when lineage advances beyond the candidate's target run."""


class ManagedIndexPublishOutcome(StrEnum):
    """Successful managed-index publisher outcomes."""

    PUBLISHED = "PUBLISHED"
    NO_INDEX_CHANGE = "NO_INDEX_CHANGE"


@dataclass(frozen=True, slots=True)
class ManagedIndexPublishResult:
    """Detached successful publication identity and planning summary."""

    outcome: ManagedIndexPublishOutcome
    index_state_id: UUID
    generation: int
    collection: str
    snapshot_hash: str
    bm25_action: BM25PlanAction
    dense_reused_count: int
    dense_computed_count: int
    document_count: int
    chunk_count: int


class ManagedStore(Protocol):
    """Physical operations required from one configured managed Qdrant adapter."""

    def create_generation_collection(self, collection_name: str) -> None: ...

    def upsert_points(
        self,
        collection_name: str,
        points: Sequence[ManagedPointInput],
        *,
        batch_size: int = 128,
    ) -> None: ...

    def retrieve_dense_vectors(
        self,
        collection_name: str,
        point_ids: Sequence[UUID],
    ) -> DenseVectorRetrievalResult: ...

    def inspect_collection(self, collection_name: str) -> ManagedCollectionMetadata: ...

    def count_points(self, collection_name: str) -> int: ...

    def validate_generation_collection(
        self,
        collection_name: str,
        expected_points: Sequence[ManagedPointInput],
    ) -> ManagedCollectionValidationResult: ...


class ManagedStoreFactory(Protocol):
    """Construct a physical adapter after the target vector dimension is known."""

    def __call__(
        self,
        managed_prefix: str,
        configuration: ManagedIndexConfiguration,
        embedding_dimension: int,
    ) -> ManagedStore: ...


@dataclass(frozen=True, slots=True)
class _MembershipSummary:
    chunk_pk: UUID
    qdrant_point_id: UUID
    dense_action: str
    dense_source_chunk_pk: UUID | None


@dataclass(frozen=True, slots=True)
class _ActiveIndexSnapshot:
    index_state_id: UUID
    generation: int
    qdrant_collection: str
    snapshot_hash: str
    embedding_fingerprint: str
    embedding_dimension: int | None
    chunking_fingerprint: str
    bm25_state: Mapping[str, Any] | None
    bm25_state_hash: str | None
    bm25_corpus_hash: str | None
    index_configuration: Mapping[str, Any] | None
    index_configuration_fingerprint: str | None
    document_count: int
    chunk_count: int
    dense_vector_name: str | None
    memberships: tuple[_MembershipSummary, ...]

    def bm25_input(self) -> PreviousBM25GenerationState:
        return PreviousBM25GenerationState(
            index_state_id=self.index_state_id,
            bm25_state=deepcopy(self.bm25_state),
            bm25_state_hash=self.bm25_state_hash,
            bm25_corpus_hash=self.bm25_corpus_hash,
            index_configuration=deepcopy(self.index_configuration),
            index_configuration_fingerprint=self.index_configuration_fingerprint,
            chunking_fingerprint=self.chunking_fingerprint,
            chunk_count=self.chunk_count,
        )

    def dense_input(self) -> PreviousDenseGenerationState:
        return PreviousDenseGenerationState(
            index_state_id=self.index_state_id,
            qdrant_collection=self.qdrant_collection,
            embedding_fingerprint=self.embedding_fingerprint,
            embedding_dimension=self.embedding_dimension,
            dense_vector_name=self.dense_vector_name,
            generation=self.generation,
        )


@dataclass(frozen=True, slots=True)
class _BuildContext:
    corpus_id: UUID
    collection_prefix: str
    active: _ActiveIndexSnapshot | None


@dataclass(frozen=True, slots=True)
class _CandidateIdentity:
    index_state_id: UUID
    generation: int
    collection: str


class _SessionCandidateReader:
    def __init__(self, session_factory: SessionFactory) -> None:
        self._session_factory = session_factory

    def find_reuse_candidates_for_hashes(
        self,
        index_state_id: UUID,
        chunk_hashes: Sequence[str],
    ) -> list[DenseReuseCandidate]:
        with self._session_factory() as session:
            return LineageRepository(session).find_reuse_candidates_for_hashes(
                index_state_id,
                chunk_hashes,
            )


class _UnavailableVectorReader:
    def retrieve_dense_vectors(
        self,
        _collection_name: str,
        _point_ids: Sequence[UUID],
    ) -> DenseVectorRetrievalResult:
        raise ManagedIndexBuildError(
            "Dense reuse was requested without a previous ACTIVE generation"
        )


class ManagedIndexService:
    """Publish one immutable hybrid generation while preserving the old ACTIVE index."""

    def __init__(
        self,
        *,
        session_factory: SessionFactory,
        embedding_provider: EmbeddingProvider,
        embedding_settings: EmbeddingModelSettings,
        chunking_settings: ChunkingSettings,
        managed_index_settings: ManagedIndexSettings,
        store_factory: ManagedStoreFactory,
        clock: Clock = utc_now,
    ) -> None:
        self._session_factory = session_factory
        self._embedding_provider = embedding_provider
        self._embedding_settings = embedding_settings
        self._chunking_settings = chunking_settings
        self._managed_index_settings = managed_index_settings
        self._store_factory = store_factory
        self._clock = clock

    def publish(
        self,
        manifest: ActiveChunkManifest,
        target_ingestion_run_id: UUID,
    ) -> ManagedIndexPublishResult:
        """Build, validate, persist, and atomically activate one desired generation."""

        desired = self._desired_configuration(manifest)
        context = self._load_build_context(manifest, target_ingestion_run_id)
        noop = self._try_no_index_change(
            manifest,
            target_ingestion_run_id,
            context,
            desired,
        )
        if noop is not None:
            return noop

        candidate = self._allocate_building_candidate(
            manifest,
            target_ingestion_run_id,
            context,
            desired,
        )
        try:
            bm25_plan = plan_bm25_generation(
                manifest,
                desired.configuration,
                context.active.bm25_input() if context.active is not None else None,
            )
            vector_reader: DenseVectorReader
            if context.active is None:
                vector_reader = _UnavailableVectorReader()
            else:
                previous_configuration = replace(
                    desired.configuration,
                    dense_vector_name=context.active.dense_vector_name or "",
                )
                vector_reader = self._store_factory(
                    context.collection_prefix,
                    previous_configuration,
                    context.active.embedding_dimension or 0,
                )
            dense_plan = DenseGenerationPlanner(
                _SessionCandidateReader(self._session_factory),
                vector_reader,
                self._embedding_provider,
            ).plan(
                manifest,
                desired.configuration.embedding_fingerprint,
                context.active.dense_input() if context.active is not None else None,
            )
            prepared = prepare_managed_generation(
                manifest,
                dense_plan,
                bm25_plan,
                PreparedIndexIdentity(
                    ingestion_run_id=target_ingestion_run_id,
                    index_state_id=candidate.index_state_id,
                    generation=candidate.generation,
                    chunking_fingerprint=manifest.chunking_fingerprint,
                    embedding_fingerprint=desired.configuration.embedding_fingerprint,
                    embedding_dimension=dense_plan.embedding_dimension,
                    point_id_format_version=(
                        desired.configuration.point_id_format_version
                    ),
                    payload_schema_version=desired.configuration.payload_schema_version,
                ),
            )
            target_store = self._store_factory(
                context.collection_prefix,
                desired.configuration,
                dense_plan.embedding_dimension,
            )
            target_store.create_generation_collection(candidate.collection)
            target_store.upsert_points(candidate.collection, prepared.points)
            target_store.validate_generation_collection(
                candidate.collection,
                prepared.points,
            )
            self._persist_ready(
                candidate,
                prepared.memberships,
                dense_plan.embedding_dimension,
                manifest,
                bm25_plan.state,
                bm25_plan.state_hash,
                bm25_plan.corpus_hash,
                desired,
            )
        except Exception as exc:
            self._mark_failed_preserving_original(
                candidate.index_state_id,
                "BUILDING",
                exc,
            )
            raise ManagedIndexBuildError(
                f"Managed index generation {candidate.generation} failed before READY"
            ) from exc

        baseline_id = (
            context.active.index_state_id if context.active is not None else None
        )
        try:
            self._activate(
                candidate,
                manifest,
                target_ingestion_run_id,
                baseline_id,
                desired.fingerprint,
            )
        except ManagedIndexActivationConflictError as exc:
            self._mark_failed_preserving_original(
                candidate.index_state_id,
                "READY",
                exc,
            )
            raise

        return ManagedIndexPublishResult(
            outcome=ManagedIndexPublishOutcome.PUBLISHED,
            index_state_id=candidate.index_state_id,
            generation=candidate.generation,
            collection=candidate.collection,
            snapshot_hash=manifest.source_snapshot_hash,
            bm25_action=bm25_plan.action,
            dense_reused_count=dense_plan.reused_count,
            dense_computed_count=dense_plan.computed_count,
            document_count=manifest.document_count,
            chunk_count=manifest.chunk_count,
        )

    def _desired_configuration(
        self,
        manifest: ActiveChunkManifest,
    ) -> ManagedIndexConfigurationIdentity:
        calculated_chunking = chunking_fingerprint(
            chunk_size=self._chunking_settings.chunk_size,
            overlap=self._chunking_settings.overlap,
        )
        if calculated_chunking != manifest.chunking_fingerprint:
            raise ManagedIndexPublishError(
                "Active manifest chunking fingerprint does not match configured chunking"
            )
        if self._embedding_provider.model_name != self._embedding_settings.model:
            raise ManagedIndexPublishError(
                "Embedding provider model does not match configured embedding model"
            )
        calculated_embedding = embedding_fingerprint(
            provider=self._embedding_settings.provider,
            model=self._embedding_settings.model,
            trust_remote_code=self._embedding_settings.trust_remote_code,
            device=self._embedding_settings.device,
        )
        return managed_index_configuration_identity_from_settings(
            self._managed_index_settings,
            embedding_fingerprint=calculated_embedding,
            chunking_fingerprint=calculated_chunking,
        )

    def _load_build_context(
        self,
        manifest: ActiveChunkManifest,
        target_ingestion_run_id: UUID,
    ) -> _BuildContext:
        with self._session_factory() as session, session.begin():
            repository = LineageRepository(session)
            corpus = repository.get_corpus_by_key(manifest.corpus_key)
            if corpus is None or corpus.corpus_id != manifest.corpus_id:
                raise LineageNotFoundError(
                    f"Managed corpus does not match manifest: {manifest.corpus_key}"
                )
            self._require_current_target_run(
                repository,
                corpus.corpus_id,
                target_ingestion_run_id,
            )
            active = repository.get_active_index_state(corpus.corpus_id)
            return _BuildContext(
                corpus_id=corpus.corpus_id,
                collection_prefix=corpus.qdrant_collection_prefix,
                active=(
                    self._detach_active(repository, active)
                    if active is not None
                    else None
                ),
            )

    def _try_no_index_change(
        self,
        manifest: ActiveChunkManifest,
        target_ingestion_run_id: UUID,
        context: _BuildContext,
        desired: ManagedIndexConfigurationIdentity,
    ) -> ManagedIndexPublishResult | None:
        active = context.active
        if active is None or not _same_logical_index(active, manifest, desired):
            return None
        try:
            bm25_plan = plan_bm25_generation(
                manifest,
                desired.configuration,
                active.bm25_input(),
            )
            if bm25_plan.action is not BM25PlanAction.REUSE:
                return None
            if len(active.memberships) != manifest.chunk_count:
                return None
            store = self._store_factory(
                context.collection_prefix,
                desired.configuration,
                active.embedding_dimension or 0,
            )
            if not _collection_shape_matches(
                store.inspect_collection(active.qdrant_collection),
                desired.configuration,
                active.embedding_dimension,
            ):
                return None
            if store.count_points(active.qdrant_collection) != manifest.chunk_count:
                return None
        except Exception:
            return None

        self._confirm_noop_baseline(
            context.corpus_id,
            active.index_state_id,
            target_ingestion_run_id,
        )
        reused = sum(
            membership.dense_action == "REUSED" for membership in active.memberships
        )
        computed = sum(
            membership.dense_action == "COMPUTED"
            for membership in active.memberships
        )
        return ManagedIndexPublishResult(
            outcome=ManagedIndexPublishOutcome.NO_INDEX_CHANGE,
            index_state_id=active.index_state_id,
            generation=active.generation,
            collection=active.qdrant_collection,
            snapshot_hash=active.snapshot_hash,
            bm25_action=bm25_plan.action,
            dense_reused_count=reused,
            dense_computed_count=computed,
            document_count=active.document_count,
            chunk_count=active.chunk_count,
        )

    def _confirm_noop_baseline(
        self,
        corpus_id: UUID,
        baseline_active_id: UUID,
        target_ingestion_run_id: UUID,
    ) -> None:
        with self._session_factory() as session, session.begin():
            repository = LineageRepository(session)
            repository.lock_corpus(corpus_id)
            current = repository.get_active_index_state(corpus_id)
            if current is None or current.index_state_id != baseline_active_id:
                raise ManagedIndexActivationConflictError(
                    "ACTIVE generation changed during no-index-change validation"
                )
            self._require_current_target_run(
                repository,
                corpus_id,
                target_ingestion_run_id,
            )

    def _allocate_building_candidate(
        self,
        manifest: ActiveChunkManifest,
        target_ingestion_run_id: UUID,
        context: _BuildContext,
        desired: ManagedIndexConfigurationIdentity,
    ) -> _CandidateIdentity:
        with self._session_factory() as session, session.begin():
            repository = LineageRepository(session)
            corpus = repository.lock_corpus(context.corpus_id)
            current = repository.get_active_index_state(context.corpus_id)
            expected_active_id = (
                context.active.index_state_id if context.active is not None else None
            )
            if _optional_state_id(current) != expected_active_id:
                raise ManagedIndexActivationConflictError(
                    "ACTIVE generation changed before candidate allocation"
                )
            self._require_current_target_run(
                repository,
                context.corpus_id,
                target_ingestion_run_id,
            )
            generation = repository.get_latest_generation_number(context.corpus_id) + 1
            collection = managed_generation_collection_name(
                corpus.qdrant_collection_prefix,
                generation,
            )
            candidate = repository.create_building_index_state(
                context.corpus_id,
                generation,
                target_ingestion_run_id,
                collection,
                manifest.source_snapshot_hash,
                desired.configuration.embedding_fingerprint,
                manifest.chunking_fingerprint,
                document_count=manifest.document_count,
                chunk_count=manifest.chunk_count,
            )
            return _CandidateIdentity(
                index_state_id=candidate.index_state_id,
                generation=candidate.generation,
                collection=candidate.qdrant_collection,
            )

    def _persist_ready(
        self,
        candidate: _CandidateIdentity,
        memberships: tuple[PreparedIndexMembership, ...],
        embedding_dimension: int,
        manifest: ActiveChunkManifest,
        bm25_state: BM25State,
        bm25_state_hash: str,
        bm25_corpus_hash: str,
        desired: ManagedIndexConfigurationIdentity,
    ) -> None:
        if len(memberships) != manifest.chunk_count:
            raise ManagedIndexBuildError(
                "Prepared membership count does not match manifest chunk count"
            )
        if len({item.chunk_pk for item in memberships}) != len(memberships):
            raise ManagedIndexBuildError("Prepared memberships contain duplicate chunks")
        if len({item.qdrant_point_id for item in memberships}) != len(memberships):
            raise ManagedIndexBuildError(
                "Prepared memberships contain duplicate Qdrant points"
            )
        with self._session_factory() as session, session.begin():
            repository = LineageRepository(session)
            state = repository.get_index_state(candidate.index_state_id)
            if state is None or state.status != "BUILDING":
                raise ManagedIndexBuildError(
                    "Candidate is no longer BUILDING during READY persistence"
                )
            repository.update_index_build_metadata(
                candidate.index_state_id,
                bm25_state=bm25_state_to_json_dict(bm25_state),
                bm25_state_hash=bm25_state_hash,
                bm25_corpus_hash=bm25_corpus_hash,
                index_configuration=desired.payload,
                index_configuration_fingerprint=desired.fingerprint,
                embedding_dimension=embedding_dimension,
                document_count=manifest.document_count,
                chunk_count=manifest.chunk_count,
            )
            repository.record_index_state_chunks_batch(
                candidate.index_state_id,
                memberships,
            )
            repository.transition_index_state(
                candidate.index_state_id,
                "BUILDING",
                "READY",
            )

    def _activate(
        self,
        candidate: _CandidateIdentity,
        manifest: ActiveChunkManifest,
        target_ingestion_run_id: UUID,
        baseline_active_id: UUID | None,
        configuration_fingerprint: str,
    ) -> None:
        with self._session_factory() as session, session.begin():
            repository = LineageRepository(session)
            repository.lock_corpus(manifest.corpus_id)
            state = repository.get_index_state(candidate.index_state_id)
            if state is None or state.status != "READY":
                raise ManagedIndexActivationConflictError(
                    "Candidate is no longer READY during activation"
                )
            if state.snapshot_hash != manifest.source_snapshot_hash:
                raise ManagedIndexActivationConflictError(
                    "Candidate snapshot identity changed before activation"
                )
            if state.ingestion_run_id != target_ingestion_run_id:
                raise ManagedIndexActivationConflictError(
                    "Candidate ingestion-run provenance changed before activation"
                )
            if (
                state.generation != candidate.generation
                or state.qdrant_collection != candidate.collection
            ):
                raise ManagedIndexActivationConflictError(
                    "Candidate physical identity changed before activation"
                )
            if state.index_configuration_fingerprint != configuration_fingerprint:
                raise ManagedIndexActivationConflictError(
                    "Candidate configuration changed before activation"
                )
            current = repository.get_active_index_state(manifest.corpus_id)
            if _optional_state_id(current) != baseline_active_id:
                raise ManagedIndexActivationConflictError(
                    "A different ACTIVE generation replaced the build baseline"
                )
            try:
                self._require_current_target_run(
                    repository,
                    manifest.corpus_id,
                    target_ingestion_run_id,
                )
            except ManagedIndexStaleLineageError:
                raise
            if current is not None:
                repository.transition_index_state(
                    current.index_state_id,
                    "ACTIVE",
                    "RETIRED",
                    occurred_at=self._clock(),
                )
            repository.transition_index_state(
                candidate.index_state_id,
                "READY",
                "ACTIVE",
                occurred_at=self._clock(),
            )

    def _require_current_target_run(
        self,
        repository: LineageRepository,
        corpus_id: UUID,
        target_ingestion_run_id: UUID,
    ) -> None:
        target = repository.get_ingestion_run(target_ingestion_run_id)
        if (
            target is None
            or target.corpus_id != corpus_id
            or target.status not in FINISHED_INGESTION_STATUSES
        ):
            raise ManagedIndexStaleLineageError(
                "Target ingestion run is not a successful/no-op run for this corpus"
            )
        latest = repository.get_latest_successful_ingestion_run(corpus_id)
        if latest is None or latest.run_id != target_ingestion_run_id:
            raise ManagedIndexStaleLineageError(
                "Lineage advanced beyond the managed-index target ingestion run"
            )

    def _mark_failed_preserving_original(
        self,
        index_state_id: UUID,
        expected_status: str,
        original: Exception,
    ) -> None:
        try:
            with self._session_factory() as session, session.begin():
                LineageRepository(session).mark_index_state_failed(
                    index_state_id,
                    expected_status,
                    str(original) or type(original).__name__,
                    self._clock(),
                )
        except Exception as failure_recording_error:
            original.add_note(
                "Additionally failed to persist candidate FAILED state: "
                f"{failure_recording_error}"
            )

    @staticmethod
    def _detach_active(
        repository: LineageRepository,
        state: IndexState,
    ) -> _ActiveIndexSnapshot:
        dense_vector_name: str | None = None
        configuration = state.index_configuration
        if isinstance(configuration, Mapping):
            dense = configuration.get("dense")
            if isinstance(dense, Mapping):
                raw_name = dense.get("vector_name")
                if isinstance(raw_name, str):
                    dense_vector_name = raw_name
        memberships = tuple(
            _MembershipSummary(
                chunk_pk=item.chunk_pk,
                qdrant_point_id=item.qdrant_point_id,
                dense_action=item.dense_action,
                dense_source_chunk_pk=item.dense_source_chunk_pk,
            )
            for item in repository.list_index_state_chunks(state.index_state_id)
        )
        return _ActiveIndexSnapshot(
            index_state_id=state.index_state_id,
            generation=state.generation,
            qdrant_collection=state.qdrant_collection,
            snapshot_hash=state.snapshot_hash,
            embedding_fingerprint=state.embedding_fingerprint,
            embedding_dimension=state.embedding_dimension,
            chunking_fingerprint=state.chunking_fingerprint,
            bm25_state=deepcopy(state.bm25_state),
            bm25_state_hash=state.bm25_state_hash,
            bm25_corpus_hash=state.bm25_corpus_hash,
            index_configuration=deepcopy(state.index_configuration),
            index_configuration_fingerprint=state.index_configuration_fingerprint,
            document_count=state.document_count,
            chunk_count=state.chunk_count,
            dense_vector_name=dense_vector_name,
            memberships=memberships,
        )


def _optional_state_id(state: IndexState | None) -> UUID | None:
    return state.index_state_id if state is not None else None


def _same_logical_index(
    active: _ActiveIndexSnapshot,
    manifest: ActiveChunkManifest,
    desired: ManagedIndexConfigurationIdentity,
) -> bool:
    return (
        active.snapshot_hash == manifest.source_snapshot_hash
        and active.chunking_fingerprint == manifest.chunking_fingerprint
        and active.embedding_fingerprint
        == desired.configuration.embedding_fingerprint
        and active.index_configuration_fingerprint == desired.fingerprint
        and active.document_count == manifest.document_count
        and active.chunk_count == manifest.chunk_count
        and active.embedding_dimension is not None
        and active.embedding_dimension > 0
        and active.bm25_state is not None
        and active.bm25_state_hash is not None
        and active.bm25_corpus_hash is not None
        and active.index_configuration is not None
    )


def _collection_shape_matches(
    metadata: ManagedCollectionMetadata,
    configuration: ManagedIndexConfiguration,
    embedding_dimension: int | None,
) -> bool:
    if embedding_dimension is None or embedding_dimension <= 0:
        return False
    return (
        metadata.dense_vector_names == (configuration.dense_vector_name,)
        and metadata.sparse_vector_names == (configuration.sparse_vector_name,)
        and metadata.dense_dimension == embedding_dimension
        and metadata.distance_metric == configuration.distance_metric.lower()
        and (
            metadata.hnsw_m is None or metadata.hnsw_m == configuration.hnsw_m
        )
        and (
            metadata.hnsw_ef_construct is None
            or metadata.hnsw_ef_construct == configuration.hnsw_ef_construct
        )
        and metadata.sparse_modifier == configuration.sparse_modifier.lower()
        and (
            metadata.sparse_on_disk is None
            or metadata.sparse_on_disk == configuration.sparse_on_disk
        )
    )
