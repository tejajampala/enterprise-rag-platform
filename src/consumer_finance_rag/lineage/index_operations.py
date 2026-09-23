"""Operational inspection and guarded cleanup for immutable index generations."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any, cast
from uuid import UUID

from qdrant_client import QdrantClient
from sqlalchemy.orm import Session

from consumer_finance_rag.access import access_policy_from_payload
from consumer_finance_rag.config import ManagedIndexSettings
from consumer_finance_rag.lineage.fingerprints import (
    BM25_IMPLEMENTATION_VERSION,
    BM25_TOKENIZER_VERSION,
    PAYLOAD_SCHEMA_VERSION,
    POINT_ID_FORMAT_VERSION,
    ManagedIndexConfiguration,
)
from consumer_finance_rag.lineage.index_types import managed_point_id
from consumer_finance_rag.lineage.managed_retrieval import (
    CorruptActiveManagedIndexError,
    parse_managed_index_configuration,
    validate_active_managed_index_metadata,
)
from consumer_finance_rag.lineage.managed_vector_store import (
    ManagedQdrantError,
    ManagedQdrantStore,
    managed_generation_collection_name,
)
from consumer_finance_rag.lineage.models import Corpus, IndexState, utc_now
from consumer_finance_rag.lineage.repository import LineageRepository
from consumer_finance_rag.pii import DETERMINISTIC_PII_VERSION, PIIType
from consumer_finance_rag.sparse import bm25_corpus_hash

SessionFactory = Callable[[], Session]
Clock = Callable[[], datetime]


class ManagedIndexOperationsError(RuntimeError):
    """Base error for managed generation operational workflows."""


class ManagedGenerationValidationError(ManagedIndexOperationsError):
    """Raised when generation validation cannot be completed safely."""


class UnsafeGenerationDeletionError(ManagedIndexOperationsError):
    """Raised when a physical generation deletion violates safety rules."""


class CleanupClassification(StrEnum):
    """Explicit cleanup disposition for a persisted or orphan generation."""

    ACTIVE_PROTECTED = "ACTIVE_PROTECTED"
    READY_PROTECTED = "READY_PROTECTED"
    BUILDING_PROTECTED = "BUILDING_PROTECTED"
    RESTORATION_SOURCE_PROTECTED = "RESTORATION_SOURCE_PROTECTED"
    RETAINED = "RETAINED"
    RECENT = "RECENT"
    ELIGIBLE_RETIRED = "ELIGIBLE_RETIRED"
    ELIGIBLE_FAILED = "ELIGIBLE_FAILED"
    PHYSICAL_MISSING = "PHYSICAL_MISSING"
    ORPHAN = "ORPHAN"
    OTHER_PROTECTED = "OTHER_PROTECTED"


class GenerationValidationClassification(StrEnum):
    """Meaning of one generation result without conflating history with serving."""

    PASS = "PASS"
    FAIL = "FAIL"
    HISTORICAL_SCHEMA_INCOMPATIBLE = "HISTORICAL_SCHEMA_INCOMPATIBLE"
    HISTORICAL_PHYSICAL_CLEANED = "HISTORICAL_PHYSICAL_CLEANED"
    FAILED_BUILD_INCOMPLETE = "FAILED_BUILD_INCOMPLETE"


class ManagedValidationStatus(StrEnum):
    """Corpus validation status with ACTIVE health as the failure boundary."""

    PASS = "PASS"
    PASS_WITH_WARNINGS = "PASS_WITH_WARNINGS"
    FAIL = "FAIL"


ELIGIBLE_CLEANUP_CLASSIFICATIONS = frozenset(
    {
        CleanupClassification.ELIGIBLE_RETIRED,
        CleanupClassification.ELIGIBLE_FAILED,
    }
)


@dataclass(frozen=True, slots=True)
class GenerationCleanupState:
    """Detached authoritative state consumed by the pure cleanup planner."""

    index_state_id: UUID
    generation: int
    status: str
    qdrant_collection: str
    retired_at: datetime | None
    failed_at: datetime | None
    restored_from_index_state_id: UUID | None = None


@dataclass(frozen=True, slots=True)
class GenerationCleanupCandidate:
    """One deterministic cleanup classification, including non-deletable entries."""

    index_state_id: UUID | None
    generation: int
    status: str
    qdrant_collection: str
    physical_present: bool
    classification: CleanupClassification
    reason: str

    @property
    def eligible(self) -> bool:
        """Return whether confirmed cleanup may re-check this physical collection."""

        return self.classification in ELIGIBLE_CLEANUP_CLASSIFICATIONS


@dataclass(frozen=True, slots=True)
class GenerationCleanupPlan:
    """Read-only cleanup plan over PostgreSQL history and Qdrant presence."""

    corpus_key: str
    managed_prefix: str
    planned_at: datetime
    retained_retired_generations: int
    retired_cleanup_min_age_hours: float
    candidates: tuple[GenerationCleanupCandidate, ...]

    @property
    def eligible(self) -> tuple[GenerationCleanupCandidate, ...]:
        return tuple(candidate for candidate in self.candidates if candidate.eligible)

    @property
    def orphans(self) -> tuple[GenerationCleanupCandidate, ...]:
        return tuple(
            candidate
            for candidate in self.candidates
            if candidate.classification is CleanupClassification.ORPHAN
        )


@dataclass(frozen=True, slots=True)
class ManagedGenerationStatus:
    """Detached generation status safe for deterministic operator output."""

    index_state_id: UUID
    generation: int
    status: str
    qdrant_collection: str
    physical_present: bool
    ingestion_run_id: UUID
    document_count: int
    chunk_count: int
    embedding_dimension: int | None
    created_at: datetime
    activated_at: datetime | None
    retired_at: datetime | None
    failed_at: datetime | None
    error_message: str | None
    restored_from_index_state_id: UUID | None
    restoration_reason: str | None
    cleanup_classification: CleanupClassification
    cleanup_reason: str


@dataclass(frozen=True, slots=True)
class ManagedGenerationValidationResult:
    """Bounded or deep validation outcome for one persisted generation."""

    index_state_id: UUID
    generation: int
    status: str
    qdrant_collection: str
    physical_present: bool
    deep: bool
    valid: bool
    issues: tuple[str, ...]
    notes: tuple[str, ...]
    classification: GenerationValidationClassification = (
        GenerationValidationClassification.PASS
    )

    @property
    def outcome(self) -> str:
        """Return the operator-facing outcome for this generation."""

        if self.classification is GenerationValidationClassification.PASS:
            return "PASS" if self.valid else "FAIL"
        if self.classification is GenerationValidationClassification.FAIL:
            return "FAIL"
        return "WARNING"


@dataclass(frozen=True, slots=True)
class ManagedGenerationValidationReport:
    """Corpus-level validation result with exactly-one-ACTIVE enforcement."""

    corpus_key: str
    active_count: int
    valid: bool
    issues: tuple[str, ...]
    generations: tuple[ManagedGenerationValidationResult, ...]

    @property
    def active_health(self) -> ManagedValidationStatus:
        """Report whether the unique ACTIVE generation is safe to serve."""

        return ManagedValidationStatus.PASS if self.valid else ManagedValidationStatus.FAIL

    @property
    def historical_findings(self) -> tuple[ManagedGenerationValidationResult, ...]:
        """Return visible non-ACTIVE warnings and integrity findings."""

        return tuple(
            result
            for result in self.generations
            if result.status != "ACTIVE"
            and (
                result.classification is not GenerationValidationClassification.PASS
                or not result.valid
                or result.issues
                or result.notes
            )
        )

    @property
    def historical_audit(self) -> str:
        """Return the compact dashboard/CLI historical audit outcome."""

        return "WARNINGS" if self.historical_findings else "PASS"

    @property
    def overall_status(self) -> ManagedValidationStatus:
        """Keep ACTIVE failures critical while retaining historical visibility."""

        if not self.valid:
            return ManagedValidationStatus.FAIL
        if self.historical_findings:
            return ManagedValidationStatus.PASS_WITH_WARNINGS
        return ManagedValidationStatus.PASS


@dataclass(frozen=True, slots=True)
class CleanupExecutionItem:
    """Outcome for one candidate re-checked immediately before deletion."""

    generation: int
    qdrant_collection: str
    deleted: bool
    reason: str


@dataclass(frozen=True, slots=True)
class GenerationCleanupExecutionResult:
    """Physical cleanup outcomes; no PostgreSQL row is ever removed."""

    confirmed: bool
    items: tuple[CleanupExecutionItem, ...]

    @property
    def deleted_collections(self) -> tuple[str, ...]:
        return tuple(item.qdrant_collection for item in self.items if item.deleted)


def plan_generation_cleanup(
    states: Sequence[GenerationCleanupState],
    *,
    corpus_key: str,
    managed_prefix: str,
    physical_collections: Sequence[str],
    retained_retired_generations: int,
    retired_cleanup_min_age_hours: float,
    now: datetime,
) -> GenerationCleanupPlan:
    """Classify cleanup eligibility without mutating PostgreSQL or Qdrant."""

    if retained_retired_generations < 0:
        raise ValueError("retained_retired_generations cannot be negative")
    if retired_cleanup_min_age_hours < 0:
        raise ValueError("retired_cleanup_min_age_hours cannot be negative")
    current_time = _aware_utc(now)
    physical = frozenset(physical_collections)
    ordered_states = tuple(sorted(states, key=lambda item: item.generation))
    newest_retired = {
        state.index_state_id
        for state in sorted(
            (
                state
                for state in ordered_states
                if state.status == "RETIRED"
                and state.qdrant_collection in physical
            ),
            key=lambda item: item.generation,
            reverse=True,
        )[:retained_retired_generations]
    }
    in_progress_restoration_sources = {
        state.restored_from_index_state_id
        for state in ordered_states
        if state.status in {"BUILDING", "READY"}
        and state.restored_from_index_state_id is not None
    }
    candidates: list[GenerationCleanupCandidate] = []
    registered_collections = {state.qdrant_collection for state in ordered_states}

    for state in ordered_states:
        present = state.qdrant_collection in physical
        canonical = _is_canonical_collection(
            managed_prefix,
            state.generation,
            state.qdrant_collection,
        )
        classification, reason = _classify_generation(
            state,
            physical_present=present,
            canonical=canonical,
            retained=state.index_state_id in newest_retired,
            restoration_source=(
                state.index_state_id in in_progress_restoration_sources
            ),
            minimum_age=timedelta(hours=retired_cleanup_min_age_hours),
            now=current_time,
        )
        candidates.append(
            GenerationCleanupCandidate(
                index_state_id=state.index_state_id,
                generation=state.generation,
                status=state.status,
                qdrant_collection=state.qdrant_collection,
                physical_present=present,
                classification=classification,
                reason=reason,
            )
        )

    for collection in sorted(physical - registered_collections):
        generation = _canonical_generation_or_none(managed_prefix, collection)
        if generation is None:
            continue
        candidates.append(
            GenerationCleanupCandidate(
                index_state_id=None,
                generation=generation,
                status="ORPHAN",
                qdrant_collection=collection,
                physical_present=True,
                classification=CleanupClassification.ORPHAN,
                reason="canonical managed collection has no PostgreSQL index_states row",
            )
        )

    return GenerationCleanupPlan(
        corpus_key=corpus_key,
        managed_prefix=managed_prefix,
        planned_at=current_time,
        retained_retired_generations=retained_retired_generations,
        retired_cleanup_min_age_hours=retired_cleanup_min_age_hours,
        candidates=tuple(
            sorted(
                candidates,
                key=lambda item: (item.generation, item.qdrant_collection),
            )
        ),
    )


class ManagedIndexOperationsService:
    """PostgreSQL-authoritative status, validation, and guarded physical cleanup."""

    def __init__(
        self,
        *,
        session_factory: SessionFactory,
        qdrant_client: QdrantClient,
        corpus_key: str,
        settings: ManagedIndexSettings,
        clock: Clock = utc_now,
    ) -> None:
        self._session_factory = session_factory
        self._client = qdrant_client
        self._corpus_key = corpus_key
        self._settings = settings
        self._clock = clock

    def cleanup_plan(self) -> GenerationCleanupPlan:
        """Read both planes and return a deterministic, mutation-free plan."""

        with self._session_factory() as session:
            corpus, states = self._load_states(session)
        return plan_generation_cleanup(
            tuple(_cleanup_state(state) for state in states),
            corpus_key=corpus.corpus_key,
            managed_prefix=corpus.qdrant_collection_prefix,
            physical_collections=self._physical_collections(),
            retained_retired_generations=(
                self._settings.retained_retired_generations
            ),
            retired_cleanup_min_age_hours=(
                self._settings.retired_cleanup_min_age_hours
            ),
            now=self._clock(),
        )

    def status(self) -> tuple[ManagedGenerationStatus, ...]:
        """Return all persisted generations with physical and cleanup state."""

        plan = self.cleanup_plan()
        planned = {
            candidate.index_state_id: candidate
            for candidate in plan.candidates
            if candidate.index_state_id is not None
        }
        with self._session_factory() as session:
            _, states = self._load_states(session)
            return tuple(
                _status_from_state(state, planned[state.index_state_id])
                for state in states
            )

    def validate(self, *, deep: bool = False) -> ManagedGenerationValidationReport:
        """Validate ACTIVE invariants and bounded historical physical state."""

        with self._session_factory() as session:
            corpus, states = self._load_states(session)
            active_count = sum(state.status == "ACTIVE" for state in states)
            report_issues: list[str] = []
            if active_count != 1:
                report_issues.append(
                    f"expected exactly one ACTIVE generation, found {active_count}"
                )
            results = tuple(
                self._validate_generation(
                    session,
                    state,
                    managed_prefix=corpus.qdrant_collection_prefix,
                    deep=deep,
                )
                for state in states
            )
        active_results = tuple(result for result in results if result.status == "ACTIVE")
        return ManagedGenerationValidationReport(
            corpus_key=corpus.corpus_key,
            active_count=active_count,
            valid=(
                not report_issues
                and len(active_results) == 1
                and active_results[0].valid
            ),
            issues=tuple(report_issues),
            generations=results,
        )

    def validate_generation(
        self,
        generation: int,
        *,
        deep: bool = False,
    ) -> ManagedGenerationValidationResult:
        """Validate one persisted generation with the shared operations checks."""

        with self._session_factory() as session:
            corpus, states = self._load_states(session)
            state = next(
                (item for item in states if item.generation == generation),
                None,
            )
            if state is None:
                raise ManagedGenerationValidationError(
                    f"Managed generation g{generation:06d} does not exist"
                )
            return self._validate_generation(
                session,
                state,
                managed_prefix=corpus.qdrant_collection_prefix,
                deep=deep,
            )

    def execute_cleanup(
        self,
        plan: GenerationCleanupPlan,
        *,
        confirmed: bool,
    ) -> GenerationCleanupExecutionResult:
        """Re-check every planned candidate under the corpus lock before deleting."""

        if plan.corpus_key != self._corpus_key:
            raise UnsafeGenerationDeletionError(
                "Cleanup plan corpus does not match this operations service"
            )
        if not confirmed:
            return GenerationCleanupExecutionResult(
                confirmed=False,
                items=tuple(
                    CleanupExecutionItem(
                        generation=candidate.generation,
                        qdrant_collection=candidate.qdrant_collection,
                        deleted=False,
                        reason="explicit --confirm was not supplied",
                    )
                    for candidate in plan.eligible
                ),
            )

        outcomes: list[CleanupExecutionItem] = []
        for original in plan.eligible:
            outcomes.append(self._execute_one(original))
        return GenerationCleanupExecutionResult(confirmed=True, items=tuple(outcomes))

    def _execute_one(
        self,
        original: GenerationCleanupCandidate,
    ) -> CleanupExecutionItem:
        if original.index_state_id is None:
            return CleanupExecutionItem(
                generation=original.generation,
                qdrant_collection=original.qdrant_collection,
                deleted=False,
                reason="orphan deletion is not supported by this milestone",
            )
        try:
            with self._session_factory() as session, session.begin():
                repository = LineageRepository(session)
                corpus = repository.get_corpus_by_key(self._corpus_key)
                if corpus is None:
                    raise UnsafeGenerationDeletionError(
                        f"Managed corpus {self._corpus_key!r} no longer exists"
                    )
                repository.lock_corpus(corpus.corpus_id)
                states = repository.list_index_states(corpus.corpus_id)
                current_plan = plan_generation_cleanup(
                    tuple(_cleanup_state(state) for state in states),
                    corpus_key=corpus.corpus_key,
                    managed_prefix=corpus.qdrant_collection_prefix,
                    physical_collections=self._physical_collections(),
                    retained_retired_generations=(
                        self._settings.retained_retired_generations
                    ),
                    retired_cleanup_min_age_hours=(
                        self._settings.retired_cleanup_min_age_hours
                    ),
                    now=self._clock(),
                )
                current = next(
                    (
                        item
                        for item in current_plan.candidates
                        if item.index_state_id == original.index_state_id
                    ),
                    None,
                )
                if current is None or not current.eligible:
                    reason = (
                        "generation state disappeared"
                        if current is None
                        else f"re-check classified generation as {current.classification.value}"
                    )
                    return CleanupExecutionItem(
                        generation=original.generation,
                        qdrant_collection=original.qdrant_collection,
                        deleted=False,
                        reason=reason,
                    )
                if current.qdrant_collection != original.qdrant_collection:
                    return CleanupExecutionItem(
                        generation=original.generation,
                        qdrant_collection=original.qdrant_collection,
                        deleted=False,
                        reason="authoritative collection identity changed",
                    )
                state = repository.get_index_state(original.index_state_id)
                if state is None or state.status not in {"RETIRED", "FAILED"}:
                    return CleanupExecutionItem(
                        generation=original.generation,
                        qdrant_collection=original.qdrant_collection,
                        deleted=False,
                        reason="authoritative status is no longer cleanup-eligible",
                    )
                store = self._store_for_state(
                    state,
                    managed_prefix=corpus.qdrant_collection_prefix,
                )
                store.delete_generation_collection(state.qdrant_collection)
        except Exception as exc:
            return CleanupExecutionItem(
                generation=original.generation,
                qdrant_collection=original.qdrant_collection,
                deleted=False,
                reason=f"physical deletion failed: {exc}",
            )
        return CleanupExecutionItem(
            generation=original.generation,
            qdrant_collection=original.qdrant_collection,
            deleted=True,
            reason="eligible physical collection deleted; PostgreSQL history retained",
        )

    def _validate_generation(
        self,
        session: Session,
        state: IndexState,
        *,
        managed_prefix: str,
        deep: bool,
    ) -> ManagedGenerationValidationResult:
        issues: list[str] = []
        notes: list[str] = []
        classification = GenerationValidationClassification.PASS
        physical_present = self._collection_exists(state.qdrant_collection)
        if not _is_canonical_collection(
            managed_prefix,
            state.generation,
            state.qdrant_collection,
        ):
            issues.append("collection name is not canonical for corpus/generation")

        repository = LineageRepository(session)
        membership_count = repository.count_index_state_chunks(state.index_state_id)
        if membership_count != state.chunk_count:
            if state.status == "FAILED":
                notes.append(
                    "FAILED build membership incomplete: "
                    f"index_state_chunks count {membership_count} of intended "
                    f"{state.chunk_count}"
                )
                classification = (
                    GenerationValidationClassification.FAILED_BUILD_INCOMPLETE
                )
            else:
                issues.append(
                    f"index_state_chunks count {membership_count} != {state.chunk_count}"
                )

        configuration: ManagedIndexConfiguration | None = None
        historical_schema_issue = (
            _historical_schema_incompatibility(state)
            if state.status == "RETIRED" and physical_present
            else None
        )
        if historical_schema_issue is not None:
            notes.append(
                "HISTORICAL_SCHEMA_INCOMPATIBLE: " + historical_schema_issue
            )
            classification = (
                GenerationValidationClassification.HISTORICAL_SCHEMA_INCOMPATIBLE
            )
        elif state.status in {"ACTIVE", "RETIRED"} and (
            state.status == "ACTIVE" or physical_present
        ):
            try:
                configuration, _ = validate_active_managed_index_metadata(state)
            except CorruptActiveManagedIndexError as exc:
                issues.append(str(exc))
            else:
                calculated_corpus_hash = bm25_corpus_hash(
                    repository.list_index_state_chunk_texts(state.index_state_id)
                )
                if calculated_corpus_hash != state.bm25_corpus_hash:
                    issues.append(
                        f"{state.status} BM25 corpus hash does not match indexed "
                        "membership texts"
                    )
        elif state.index_configuration is not None:
            try:
                configuration = parse_managed_index_configuration(
                    cast(Mapping[str, Any], state.index_configuration)
                )
            except CorruptActiveManagedIndexError as exc:
                notes.append(f"historical configuration is incomplete: {exc}")

        if state.status == "ACTIVE" and not physical_present:
            issues.append("ACTIVE Qdrant collection is missing")
        elif state.status == "RETIRED" and not physical_present:
            notes.append("physical=cleaned")
            classification = (
                GenerationValidationClassification.HISTORICAL_PHYSICAL_CLEANED
            )
        elif state.status == "FAILED" and not physical_present:
            notes.append("FAILED history has no physical collection")
            classification = GenerationValidationClassification.FAILED_BUILD_INCOMPLETE
        elif state.status in {"READY", "BUILDING"} and not physical_present:
            issues.append(f"{state.status} Qdrant collection is missing")

        if physical_present and configuration is not None:
            self._validate_physical_shape(state, configuration, managed_prefix, issues)
            if deep:
                self._deep_validate_points(
                    session,
                    state,
                    configuration,
                    managed_prefix,
                    issues,
                )
        elif physical_present and state.status == "FAILED":
            notes.append("FAILED physical collection is present and may be partial")
            classification = GenerationValidationClassification.FAILED_BUILD_INCOMPLETE
        elif physical_present and configuration is None:
            if historical_schema_issue is None:
                issues.append(
                    "physical collection cannot be validated without configuration"
                )

        if issues:
            classification = GenerationValidationClassification.FAIL

        return ManagedGenerationValidationResult(
            index_state_id=state.index_state_id,
            generation=state.generation,
            status=state.status,
            qdrant_collection=state.qdrant_collection,
            physical_present=physical_present,
            deep=deep,
            valid=classification is GenerationValidationClassification.PASS,
            issues=tuple(issues),
            notes=tuple(notes),
            classification=classification,
        )

    def _validate_physical_shape(
        self,
        state: IndexState,
        configuration: ManagedIndexConfiguration,
        managed_prefix: str,
        issues: list[str],
    ) -> None:
        if state.embedding_dimension is None or state.embedding_dimension <= 0:
            issues.append("embedding dimension is missing or invalid")
            return
        try:
            store = ManagedQdrantStore(
                self._client,
                managed_prefix=managed_prefix,
                configuration=configuration,
                embedding_dimension=state.embedding_dimension,
            )
            metadata = store.inspect_collection(state.qdrant_collection)
            point_count = store.count_points(state.qdrant_collection)
        except Exception as exc:
            issues.append(f"collection is not readable: {exc}")
            return
        if configuration.dense_vector_name not in metadata.dense_vector_names:
            issues.append(f"missing dense vector {configuration.dense_vector_name!r}")
        if configuration.sparse_vector_name not in metadata.sparse_vector_names:
            issues.append(f"missing sparse vector {configuration.sparse_vector_name!r}")
        if metadata.dense_dimension != state.embedding_dimension:
            issues.append(
                f"dense dimension {metadata.dense_dimension} != {state.embedding_dimension}"
            )
        if metadata.distance_metric != configuration.distance_metric.lower():
            issues.append(
                f"distance {metadata.distance_metric!r} != "
                f"{configuration.distance_metric.lower()!r}"
            )
        if metadata.hnsw_m is not None and metadata.hnsw_m != configuration.hnsw_m:
            issues.append(f"HNSW m {metadata.hnsw_m} != {configuration.hnsw_m}")
        if (
            metadata.hnsw_ef_construct is not None
            and metadata.hnsw_ef_construct != configuration.hnsw_ef_construct
        ):
            issues.append(
                f"HNSW ef_construct {metadata.hnsw_ef_construct} != "
                f"{configuration.hnsw_ef_construct}"
            )
        if metadata.sparse_modifier != configuration.sparse_modifier.lower():
            issues.append(
                f"sparse modifier {metadata.sparse_modifier!r} != "
                f"{configuration.sparse_modifier.lower()!r}"
            )
        if (
            metadata.sparse_on_disk is not None
            and metadata.sparse_on_disk != configuration.sparse_on_disk
        ):
            issues.append(
                f"sparse on_disk {metadata.sparse_on_disk} != "
                f"{configuration.sparse_on_disk}"
            )
        if point_count != state.chunk_count:
            issues.append(f"Qdrant point count {point_count} != {state.chunk_count}")
        if configuration.payload_schema_version != PAYLOAD_SCHEMA_VERSION:
            issues.append("payload schema is unsupported by the current managed runtime")
        if configuration.pii_detector_version != DETERMINISTIC_PII_VERSION:
            issues.append("PII payload schema/detector is unsupported by current runtime")

    def _deep_validate_points(
        self,
        session: Session,
        state: IndexState,
        configuration: ManagedIndexConfiguration,
        managed_prefix: str,
        issues: list[str],
    ) -> None:
        if state.embedding_dimension is None:
            return
        store = ManagedQdrantStore(
            self._client,
            managed_prefix=managed_prefix,
            configuration=configuration,
            embedding_dimension=state.embedding_dimension,
        )
        try:
            points = store.scroll_all_points(state.qdrant_collection)
        except Exception as exc:
            issues.append(f"deep collection scan failed: {exc}")
            return
        memberships = LineageRepository(session).list_index_state_chunks(
            state.index_state_id
        )
        expected = {item.qdrant_point_id: item.chunk_pk for item in memberships}
        actual = {point.point_id: point for point in points}
        missing = sorted(set(expected) - set(actual), key=str)
        unexpected = sorted(set(actual) - set(expected), key=str)
        if missing:
            issues.append("deep validation missing point IDs: " + ", ".join(map(str, missing)))
        if unexpected:
            issues.append(
                "deep validation unexpected point IDs: " + ", ".join(map(str, unexpected))
            )
        for point_id in sorted(set(expected) & set(actual), key=str):
            point = actual[point_id]
            chunk_pk = expected[point_id]
            if point_id != managed_point_id(chunk_pk):
                issues.append(f"point {point_id} does not match lineage chunk identity")
            self._validate_deep_payload(
                point_id,
                point.payload,
                state,
                chunk_pk,
                configuration,
                issues,
            )
            if point.dense_vector is None or len(point.dense_vector) != state.embedding_dimension:
                issues.append(f"point {point_id} dense vector is missing or incompatible")
            if point.sparse_vector is None:
                issues.append(f"point {point_id} sparse vector is missing")

    @staticmethod
    def _validate_deep_payload(
        point_id: UUID,
        payload: Mapping[str, Any],
        state: IndexState,
        chunk_pk: UUID,
        configuration: ManagedIndexConfiguration,
        issues: list[str],
    ) -> None:
        expected = {
            "index_generation": state.generation,
            "index_state_id": str(state.index_state_id),
            "lineage_chunk_pk": str(chunk_pk),
            "chunking_fingerprint": state.chunking_fingerprint,
            "payload_schema_version": PAYLOAD_SCHEMA_VERSION,
            "point_id_format_version": configuration.point_id_format_version,
        }
        for key, value in expected.items():
            if payload.get(key) != value:
                issues.append(f"point {point_id} payload {key!r} mismatch")
        for key in (
            "document_id",
            "chunk_id",
            "document_version",
            "title",
            "source_id",
            "source_key",
            "chunk_hash",
            "text",
            "content_hash",
            "source_version",
            "lineage_document_pk",
            "lineage_document_version_id",
            "ingestion_run_id",
        ):
            if not isinstance(payload.get(key), str) or not payload[key]:
                issues.append(f"point {point_id} payload {key!r} is missing or invalid")
        try:
            access_policy_from_payload(payload)
        except (TypeError, ValueError) as exc:
            issues.append(f"point {point_id} ACL payload is invalid: {exc}")
        contains_pii = payload.get("contains_pii")
        pii_types = payload.get("pii_types")
        detector = payload.get("pii_detector_version")
        if not isinstance(contains_pii, bool):
            issues.append(f"point {point_id} contains_pii must be boolean")
        if not isinstance(pii_types, list) or not all(
            isinstance(item, str) and item in {pii_type.value for pii_type in PIIType}
            for item in pii_types if isinstance(pii_types, list)
        ):
            issues.append(f"point {point_id} PII types payload is invalid")
        elif isinstance(contains_pii, bool) and contains_pii != bool(pii_types):
            issues.append(f"point {point_id} PII presence disagrees with types")
        if detector != configuration.pii_detector_version:
            issues.append(f"point {point_id} PII detector version mismatch")

    def _store_for_state(
        self,
        state: IndexState,
        *,
        managed_prefix: str,
    ) -> ManagedQdrantStore:
        configuration = _configuration_for_cleanup(state, self._settings)
        return ManagedQdrantStore(
            self._client,
            managed_prefix=managed_prefix,
            configuration=configuration,
            embedding_dimension=state.embedding_dimension or 1,
        )

    def _load_states(self, session: Session) -> tuple[Corpus, list[IndexState]]:
        repository = LineageRepository(session)
        corpus = repository.get_corpus_by_key(self._corpus_key)
        if corpus is None:
            raise ManagedIndexOperationsError(
                f"Managed corpus {self._corpus_key!r} is not initialized"
            )
        return corpus, repository.list_index_states(corpus.corpus_id)

    def _physical_collections(self) -> tuple[str, ...]:
        try:
            return tuple(
                sorted(collection.name for collection in self._client.get_collections().collections)
            )
        except Exception as exc:
            raise ManagedIndexOperationsError(
                f"Unable to list Qdrant collections: {exc}"
            ) from exc

    def _collection_exists(self, collection_name: str) -> bool:
        try:
            return bool(self._client.collection_exists(collection_name))
        except Exception as exc:
            raise ManagedGenerationValidationError(
                f"Unable to inspect Qdrant collection {collection_name!r}: {exc}"
            ) from exc


def _classify_generation(
    state: GenerationCleanupState,
    *,
    physical_present: bool,
    canonical: bool,
    retained: bool,
    restoration_source: bool,
    minimum_age: timedelta,
    now: datetime,
) -> tuple[CleanupClassification, str]:
    if state.status == "ACTIVE":
        return CleanupClassification.ACTIVE_PROTECTED, "ACTIVE is never deleted"
    if state.status == "READY":
        return CleanupClassification.READY_PROTECTED, "READY is never deleted"
    if state.status == "BUILDING":
        return CleanupClassification.BUILDING_PROTECTED, "BUILDING is never deleted"
    if restoration_source:
        return (
            CleanupClassification.RESTORATION_SOURCE_PROTECTED,
            "source physical collection is needed by an in-progress restoration",
        )
    if not physical_present:
        return CleanupClassification.PHYSICAL_MISSING, "no physical cleanup required"
    if not canonical:
        return (
            CleanupClassification.OTHER_PROTECTED,
            "collection name is not canonical for the persisted generation",
        )
    if state.status == "FAILED":
        return CleanupClassification.ELIGIBLE_FAILED, "FAILED physical collection is eligible"
    if state.status != "RETIRED":
        return CleanupClassification.OTHER_PROTECTED, f"status {state.status} is protected"
    if retained:
        return CleanupClassification.RETAINED, "within newest RETIRED retention count"
    if state.retired_at is None:
        return CleanupClassification.RECENT, "RETIRED generation has no retirement timestamp"
    retirement_time = _aware_utc(state.retired_at)
    if now - retirement_time < minimum_age:
        return CleanupClassification.RECENT, "RETIRED generation is inside minimum-age grace"
    return CleanupClassification.ELIGIBLE_RETIRED, "outside retention and minimum-age grace"


def _status_from_state(
    state: IndexState,
    candidate: GenerationCleanupCandidate,
) -> ManagedGenerationStatus:
    return ManagedGenerationStatus(
        index_state_id=state.index_state_id,
        generation=state.generation,
        status=state.status,
        qdrant_collection=state.qdrant_collection,
        physical_present=candidate.physical_present,
        ingestion_run_id=state.ingestion_run_id,
        document_count=state.document_count,
        chunk_count=state.chunk_count,
        embedding_dimension=state.embedding_dimension,
        created_at=state.created_at,
        activated_at=state.activated_at,
        retired_at=state.retired_at,
        failed_at=state.failed_at,
        error_message=state.error_message,
        restored_from_index_state_id=state.restored_from_index_state_id,
        restoration_reason=state.restoration_reason,
        cleanup_classification=candidate.classification,
        cleanup_reason=candidate.reason,
    )


def _cleanup_state(state: IndexState) -> GenerationCleanupState:
    return GenerationCleanupState(
        index_state_id=state.index_state_id,
        generation=state.generation,
        status=state.status,
        qdrant_collection=state.qdrant_collection,
        retired_at=state.retired_at,
        failed_at=state.failed_at,
        restored_from_index_state_id=state.restored_from_index_state_id,
    )


def _historical_schema_incompatibility(state: IndexState) -> str | None:
    """Describe RETIRED metadata that the current validator cannot verify safely."""

    payload = state.index_configuration
    if not isinstance(payload, Mapping):
        return "persisted configuration is absent or is not an object"
    if "pii_detector_version" not in payload:
        return "persisted configuration predates required field pii_detector_version"
    try:
        configuration = parse_managed_index_configuration(
            cast(Mapping[str, Any], payload)
        )
    except CorruptActiveManagedIndexError:
        return "persisted configuration is not supported by the current schema"

    incompatible: list[str] = []
    if configuration.payload_schema_version != PAYLOAD_SCHEMA_VERSION:
        incompatible.append("payload schema")
    if configuration.point_id_format_version != POINT_ID_FORMAT_VERSION:
        incompatible.append("point-ID format")
    if configuration.bm25_implementation_version != BM25_IMPLEMENTATION_VERSION:
        incompatible.append("BM25 implementation")
    if configuration.bm25_tokenizer_version != BM25_TOKENIZER_VERSION:
        incompatible.append("BM25 tokenizer")
    if configuration.pii_detector_version != DETERMINISTIC_PII_VERSION:
        incompatible.append("PII detector")
    if incompatible:
        return (
            "persisted configuration uses unsupported runtime version(s): "
            + ", ".join(incompatible)
        )
    return None


def _configuration_for_cleanup(
    state: IndexState,
    settings: ManagedIndexSettings,
) -> ManagedIndexConfiguration:
    if state.index_configuration is not None:
        try:
            return parse_managed_index_configuration(
                cast(Mapping[str, Any], state.index_configuration)
            )
        except CorruptActiveManagedIndexError:
            pass
    return ManagedIndexConfiguration(
        embedding_fingerprint=state.embedding_fingerprint,
        chunking_fingerprint=state.chunking_fingerprint,
        bm25_k1=settings.bm25_k1,
        bm25_b=settings.bm25_b,
        dense_vector_name=settings.dense_vector_name,
        sparse_vector_name=settings.sparse_vector_name,
        distance_metric=settings.distance_metric,
        hnsw_m=settings.hnsw_m,
        hnsw_ef_construct=settings.hnsw_ef_construct,
        sparse_modifier=settings.sparse_modifier,
        sparse_on_disk=settings.sparse_on_disk,
        managed_index_format_version=settings.format_version,
    )


def _is_canonical_collection(prefix: str, generation: int, collection: str) -> bool:
    try:
        return managed_generation_collection_name(prefix, generation) == collection
    except ManagedQdrantError:
        return False


def _canonical_generation_or_none(prefix: str, collection: str) -> int | None:
    marker = f"{prefix}_g"
    if not collection.startswith(marker):
        return None
    suffix = collection.removeprefix(marker)
    if len(suffix) < 6 or not suffix.isdigit():
        return None
    generation = int(suffix)
    return generation if _is_canonical_collection(prefix, generation, collection) else None


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
