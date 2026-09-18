"""SQLAlchemy models for Phase 3 application-level lineage."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import (
    CHAR,
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, validates

INGESTION_RUN_STATUSES = frozenset({"RUNNING", "SUCCEEDED", "FAILED", "NOOP"})
DOCUMENT_STATES = frozenset({"ACTIVE", "DELETED"})
DOCUMENT_CHANGE_TYPES = frozenset({"NEW", "CHANGED", "UNCHANGED", "DELETED"})
INDEX_STATE_STATUSES = frozenset({"BUILDING", "READY", "ACTIVE", "RETIRED", "FAILED"})
DENSE_ACTIONS = frozenset({"COMPUTED", "REUSED"})
PORTABLE_JSON = JSON().with_variant(JSONB(), "postgresql")


def utc_now() -> datetime:
    """Return an aware UTC timestamp for application-generated defaults."""

    return datetime.now(UTC)


def _validate_domain_value(attribute: str, value: str, allowed: frozenset[str]) -> str:
    if value not in allowed:
        choices = ", ".join(sorted(allowed))
        raise ValueError(f"{attribute} must be one of: {choices}")
    return value


class Base(DeclarativeBase):
    """Declarative base for the application-lineage schema only."""


class Corpus(Base):
    """An independently managed Phase 3 corpus profile."""

    __tablename__ = "corpora"

    corpus_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    corpus_key: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    qdrant_collection_prefix: Mapped[str] = mapped_column(
        String(255), nullable=False, unique=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now, server_default=func.now()
    )

    ingestion_runs: Mapped[list[IngestionRun]] = relationship(back_populates="corpus")
    documents: Mapped[list[LineageDocument]] = relationship(back_populates="corpus")
    index_states: Mapped[list[IndexState]] = relationship(back_populates="corpus")


class IngestionRun(Base):
    """One Phase 3 managed ingestion attempt."""

    __tablename__ = "ingestion_runs"
    __table_args__ = (
        CheckConstraint("documents_new >= 0", name="ck_ingestion_runs_documents_new"),
        CheckConstraint("documents_changed >= 0", name="ck_ingestion_runs_documents_changed"),
        CheckConstraint(
            "documents_unchanged >= 0", name="ck_ingestion_runs_documents_unchanged"
        ),
        CheckConstraint("documents_deleted >= 0", name="ck_ingestion_runs_documents_deleted"),
    )

    run_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    corpus_id: Mapped[UUID] = mapped_column(
        Uuid, ForeignKey("corpora.corpus_id"), nullable=False
    )
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    authoritative_snapshot: Mapped[bool] = mapped_column(Boolean, nullable=False)
    snapshot_hash: Mapped[str | None] = mapped_column(CHAR(64))
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now, server_default=func.now()
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_message: Mapped[str | None] = mapped_column(Text)
    documents_new: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )
    documents_changed: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )
    documents_unchanged: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )
    documents_deleted: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )

    corpus: Mapped[Corpus] = relationship(back_populates="ingestion_runs")
    classifications: Mapped[list[IngestionRunDocument]] = relationship(back_populates="run")
    index_states: Mapped[list[IndexState]] = relationship(back_populates="ingestion_run")

    @validates("status")
    def validate_status(self, _key: str, value: str) -> str:
        """Validate ingestion status without a database-native enum."""

        return _validate_domain_value("status", value, INGESTION_RUN_STATUSES)


class LineageDocument(Base):
    """Stable logical source-document identity across content versions."""

    __tablename__ = "documents"
    __table_args__ = (
        UniqueConstraint("corpus_id", "source_key", name="uq_documents_corpus_source_key"),
    )

    document_pk: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    corpus_id: Mapped[UUID] = mapped_column(
        Uuid, ForeignKey("corpora.corpus_id"), nullable=False
    )
    source_key: Mapped[str] = mapped_column(String(512), nullable=False)
    source_id: Mapped[str] = mapped_column(String(512), nullable=False)
    source_url: Mapped[str | None] = mapped_column(Text)
    state: Mapped[str] = mapped_column(String(32), nullable=False)
    first_seen_run_id: Mapped[UUID] = mapped_column(
        Uuid, ForeignKey("ingestion_runs.run_id"), nullable=False
    )
    last_seen_run_id: Mapped[UUID] = mapped_column(
        Uuid, ForeignKey("ingestion_runs.run_id"), nullable=False
    )
    deleted_run_id: Mapped[UUID | None] = mapped_column(
        Uuid, ForeignKey("ingestion_runs.run_id")
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        onupdate=utc_now,
        server_default=func.now(),
    )

    corpus: Mapped[Corpus] = relationship(back_populates="documents")
    first_seen_run: Mapped[IngestionRun] = relationship(foreign_keys=[first_seen_run_id])
    last_seen_run: Mapped[IngestionRun] = relationship(foreign_keys=[last_seen_run_id])
    deleted_run: Mapped[IngestionRun | None] = relationship(foreign_keys=[deleted_run_id])
    versions: Mapped[list[DocumentVersion]] = relationship(
        back_populates="document", order_by="DocumentVersion.version_number"
    )
    classifications: Mapped[list[IngestionRunDocument]] = relationship(
        back_populates="document"
    )

    @validates("state")
    def validate_state(self, _key: str, value: str) -> str:
        """Validate document state without a database-native enum."""

        return _validate_domain_value("state", value, DOCUMENT_STATES)


class DocumentVersion(Base):
    """An immutable observed version of a logical document."""

    __tablename__ = "document_versions"
    __table_args__ = (
        UniqueConstraint(
            "document_pk", "version_number", name="uq_document_versions_document_number"
        ),
        UniqueConstraint(
            "document_pk", "version_hash", name="uq_document_versions_document_hash"
        ),
        CheckConstraint("version_number > 0", name="ck_document_versions_version_positive"),
    )

    document_version_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    document_pk: Mapped[UUID] = mapped_column(
        Uuid, ForeignKey("documents.document_pk"), nullable=False
    )
    version_number: Mapped[int] = mapped_column(Integer, nullable=False)
    content_hash: Mapped[str] = mapped_column(CHAR(64), nullable=False)
    metadata_hash: Mapped[str] = mapped_column(CHAR(64), nullable=False)
    version_hash: Mapped[str] = mapped_column(CHAR(64), nullable=False)
    compatibility_document_id: Mapped[str] = mapped_column(String(1024), nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str] = mapped_column(String(255), nullable=False)
    source_url: Mapped[str | None] = mapped_column(Text)
    source_version: Mapped[str] = mapped_column(String(255), nullable=False)
    normalized_text: Mapped[str] = mapped_column(Text, nullable=False)
    created_run_id: Mapped[UUID] = mapped_column(
        Uuid, ForeignKey("ingestion_runs.run_id"), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now, server_default=func.now()
    )

    document: Mapped[LineageDocument] = relationship(back_populates="versions")
    created_run: Mapped[IngestionRun] = relationship(foreign_keys=[created_run_id])
    chunks: Mapped[list[LineageChunk]] = relationship(
        back_populates="document_version", order_by="LineageChunk.ordinal"
    )


class LineageChunk(Base):
    """An immutable chunk derived from a particular document version."""

    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint(
            "document_version_id",
            "chunking_fingerprint",
            "ordinal",
            name="uq_chunks_version_fingerprint_ordinal",
        ),
        UniqueConstraint(
            "document_version_id",
            "chunking_fingerprint",
            "compatibility_chunk_id",
            name="uq_chunks_version_fingerprint_compatibility_id",
        ),
        CheckConstraint("ordinal >= 0", name="ck_chunks_ordinal_nonnegative"),
    )

    chunk_pk: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    document_version_id: Mapped[UUID] = mapped_column(
        Uuid, ForeignKey("document_versions.document_version_id"), nullable=False
    )
    chunking_fingerprint: Mapped[str] = mapped_column(CHAR(64), nullable=False)
    compatibility_chunk_id: Mapped[str] = mapped_column(String(1024), nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_hash: Mapped[str] = mapped_column(CHAR(64), nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_metadata: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now, server_default=func.now()
    )

    document_version: Mapped[DocumentVersion] = relationship(back_populates="chunks")
    index_memberships: Mapped[list[IndexStateChunk]] = relationship(
        back_populates="chunk", foreign_keys="IndexStateChunk.chunk_pk"
    )


class IngestionRunDocument(Base):
    """Classification of one logical document during one ingestion run."""

    __tablename__ = "ingestion_run_documents"

    run_id: Mapped[UUID] = mapped_column(
        Uuid, ForeignKey("ingestion_runs.run_id"), primary_key=True
    )
    document_pk: Mapped[UUID] = mapped_column(
        Uuid, ForeignKey("documents.document_pk"), primary_key=True
    )
    change_type: Mapped[str] = mapped_column(String(32), nullable=False)
    previous_version_id: Mapped[UUID | None] = mapped_column(
        Uuid, ForeignKey("document_versions.document_version_id")
    )
    observed_version_id: Mapped[UUID | None] = mapped_column(
        Uuid, ForeignKey("document_versions.document_version_id")
    )
    content_changed: Mapped[bool] = mapped_column(Boolean, nullable=False)
    metadata_changed: Mapped[bool] = mapped_column(Boolean, nullable=False)

    run: Mapped[IngestionRun] = relationship(back_populates="classifications")
    document: Mapped[LineageDocument] = relationship(back_populates="classifications")
    previous_version: Mapped[DocumentVersion | None] = relationship(
        foreign_keys=[previous_version_id]
    )
    observed_version: Mapped[DocumentVersion | None] = relationship(
        foreign_keys=[observed_version_id]
    )

    @validates("change_type")
    def validate_change_type(self, _key: str, value: str) -> str:
        """Validate change classification without a database-native enum."""

        return _validate_domain_value("change_type", value, DOCUMENT_CHANGE_TYPES)


class IndexState(Base):
    """One physical Phase 3 Qdrant index generation."""

    __tablename__ = "index_states"
    __table_args__ = (
        UniqueConstraint("corpus_id", "generation", name="uq_index_states_corpus_generation"),
        Index(
            "uq_index_states_one_active_per_corpus",
            "corpus_id",
            unique=True,
            postgresql_where=text("status = 'ACTIVE'"),
            sqlite_where=text("status = 'ACTIVE'"),
        ),
        CheckConstraint("generation > 0", name="ck_index_states_generation_positive"),
        CheckConstraint(
            "embedding_dimension IS NULL OR embedding_dimension > 0",
            name="ck_index_states_embedding_dimension_positive",
        ),
        CheckConstraint("document_count >= 0", name="ck_index_states_document_count_nonnegative"),
        CheckConstraint("chunk_count >= 0", name="ck_index_states_chunk_count_nonnegative"),
    )

    index_state_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    corpus_id: Mapped[UUID] = mapped_column(
        Uuid, ForeignKey("corpora.corpus_id"), nullable=False
    )
    generation: Mapped[int] = mapped_column(Integer, nullable=False)
    ingestion_run_id: Mapped[UUID] = mapped_column(
        Uuid, ForeignKey("ingestion_runs.run_id"), nullable=False
    )
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    qdrant_collection: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    snapshot_hash: Mapped[str] = mapped_column(CHAR(64), nullable=False)
    bm25_corpus_hash: Mapped[str | None] = mapped_column(CHAR(64))
    bm25_state_hash: Mapped[str | None] = mapped_column(CHAR(64))
    bm25_state: Mapped[dict[str, Any] | None] = mapped_column(PORTABLE_JSON)
    index_configuration: Mapped[dict[str, Any] | None] = mapped_column(PORTABLE_JSON)
    index_configuration_fingerprint: Mapped[str | None] = mapped_column(CHAR(64))
    embedding_fingerprint: Mapped[str] = mapped_column(CHAR(64), nullable=False)
    embedding_dimension: Mapped[int | None] = mapped_column(Integer)
    chunking_fingerprint: Mapped[str] = mapped_column(CHAR(64), nullable=False)
    document_count: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now, server_default=func.now()
    )
    activated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    retired_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_message: Mapped[str | None] = mapped_column(Text)
    failed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    corpus: Mapped[Corpus] = relationship(back_populates="index_states")
    ingestion_run: Mapped[IngestionRun] = relationship(back_populates="index_states")
    chunks: Mapped[list[IndexStateChunk]] = relationship(back_populates="index_state")

    @validates("status")
    def validate_status(self, _key: str, value: str) -> str:
        """Validate index status without a database-native enum."""

        return _validate_domain_value("status", value, INDEX_STATE_STATUSES)


class IndexStateChunk(Base):
    """Membership of a lineage chunk in an immutable index generation."""

    __tablename__ = "index_state_chunks"
    __table_args__ = (
        UniqueConstraint(
            "index_state_id", "qdrant_point_id", name="uq_index_state_chunks_state_point"
        ),
        CheckConstraint(
            "(dense_action = 'COMPUTED' AND dense_source_chunk_pk IS NULL) OR "
            "(dense_action = 'REUSED' AND dense_source_chunk_pk IS NOT NULL)",
            name="ck_index_state_chunks_dense_provenance",
        ),
    )

    index_state_id: Mapped[UUID] = mapped_column(
        Uuid, ForeignKey("index_states.index_state_id"), primary_key=True
    )
    chunk_pk: Mapped[UUID] = mapped_column(Uuid, ForeignKey("chunks.chunk_pk"), primary_key=True)
    qdrant_point_id: Mapped[UUID] = mapped_column(Uuid, nullable=False)
    dense_action: Mapped[str] = mapped_column(String(32), nullable=False)
    dense_source_chunk_pk: Mapped[UUID | None] = mapped_column(
        Uuid, ForeignKey("chunks.chunk_pk")
    )
    indexed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now, server_default=func.now()
    )

    index_state: Mapped[IndexState] = relationship(back_populates="chunks")
    chunk: Mapped[LineageChunk] = relationship(
        back_populates="index_memberships", foreign_keys=[chunk_pk]
    )
    dense_source_chunk: Mapped[LineageChunk | None] = relationship(
        foreign_keys=[dense_source_chunk_pk]
    )

    @validates("dense_action")
    def validate_dense_action(self, _key: str, value: str) -> str:
        """Validate vector provenance without a database-native enum."""

        return _validate_domain_value("dense_action", value, DENSE_ACTIONS)
