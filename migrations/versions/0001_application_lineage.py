"""Create the initial Phase 3 application-lineage schema.

Revision ID: 0001_application_lineage
Revises:
Create Date: 2026-09-18
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001_application_lineage"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create only the application-lineage tables."""

    op.create_table(
        "corpora",
        sa.Column("corpus_id", sa.Uuid(), nullable=False),
        sa.Column("corpus_key", sa.String(length=255), nullable=False),
        sa.Column("qdrant_collection_prefix", sa.String(length=255), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("corpus_id"),
        sa.UniqueConstraint("corpus_key"),
        sa.UniqueConstraint("qdrant_collection_prefix"),
    )
    op.create_table(
        "ingestion_runs",
        sa.Column("run_id", sa.Uuid(), nullable=False),
        sa.Column("corpus_id", sa.Uuid(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("authoritative_snapshot", sa.Boolean(), nullable=False),
        sa.Column("snapshot_hash", sa.CHAR(length=64), nullable=True),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("documents_new", sa.Integer(), server_default="0", nullable=False),
        sa.Column("documents_changed", sa.Integer(), server_default="0", nullable=False),
        sa.Column("documents_unchanged", sa.Integer(), server_default="0", nullable=False),
        sa.Column("documents_deleted", sa.Integer(), server_default="0", nullable=False),
        sa.CheckConstraint("documents_changed >= 0", name="ck_ingestion_runs_documents_changed"),
        sa.CheckConstraint("documents_deleted >= 0", name="ck_ingestion_runs_documents_deleted"),
        sa.CheckConstraint("documents_new >= 0", name="ck_ingestion_runs_documents_new"),
        sa.CheckConstraint(
            "documents_unchanged >= 0", name="ck_ingestion_runs_documents_unchanged"
        ),
        sa.ForeignKeyConstraint(["corpus_id"], ["corpora.corpus_id"]),
        sa.PrimaryKeyConstraint("run_id"),
    )
    op.create_table(
        "documents",
        sa.Column("document_pk", sa.Uuid(), nullable=False),
        sa.Column("corpus_id", sa.Uuid(), nullable=False),
        sa.Column("source_key", sa.String(length=512), nullable=False),
        sa.Column("source_id", sa.String(length=512), nullable=False),
        sa.Column("source_url", sa.Text(), nullable=True),
        sa.Column("state", sa.String(length=32), nullable=False),
        sa.Column("first_seen_run_id", sa.Uuid(), nullable=False),
        sa.Column("last_seen_run_id", sa.Uuid(), nullable=False),
        sa.Column("deleted_run_id", sa.Uuid(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["corpus_id"], ["corpora.corpus_id"]),
        sa.ForeignKeyConstraint(["deleted_run_id"], ["ingestion_runs.run_id"]),
        sa.ForeignKeyConstraint(["first_seen_run_id"], ["ingestion_runs.run_id"]),
        sa.ForeignKeyConstraint(["last_seen_run_id"], ["ingestion_runs.run_id"]),
        sa.PrimaryKeyConstraint("document_pk"),
        sa.UniqueConstraint("corpus_id", "source_key", name="uq_documents_corpus_source_key"),
    )
    op.create_table(
        "document_versions",
        sa.Column("document_version_id", sa.Uuid(), nullable=False),
        sa.Column("document_pk", sa.Uuid(), nullable=False),
        sa.Column("version_number", sa.Integer(), nullable=False),
        sa.Column("content_hash", sa.CHAR(length=64), nullable=False),
        sa.Column("metadata_hash", sa.CHAR(length=64), nullable=False),
        sa.Column("version_hash", sa.CHAR(length=64), nullable=False),
        sa.Column("compatibility_document_id", sa.String(length=1024), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("category", sa.String(length=255), nullable=False),
        sa.Column("source_url", sa.Text(), nullable=True),
        sa.Column("source_version", sa.String(length=255), nullable=False),
        sa.Column("normalized_text", sa.Text(), nullable=False),
        sa.Column("created_run_id", sa.Uuid(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "version_number > 0", name="ck_document_versions_version_positive"
        ),
        sa.ForeignKeyConstraint(["created_run_id"], ["ingestion_runs.run_id"]),
        sa.ForeignKeyConstraint(["document_pk"], ["documents.document_pk"]),
        sa.PrimaryKeyConstraint("document_version_id"),
        sa.UniqueConstraint(
            "document_pk", "version_hash", name="uq_document_versions_document_hash"
        ),
        sa.UniqueConstraint(
            "document_pk", "version_number", name="uq_document_versions_document_number"
        ),
    )
    op.create_table(
        "chunks",
        sa.Column("chunk_pk", sa.Uuid(), nullable=False),
        sa.Column("document_version_id", sa.Uuid(), nullable=False),
        sa.Column("chunking_fingerprint", sa.CHAR(length=64), nullable=False),
        sa.Column("compatibility_chunk_id", sa.String(length=1024), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("chunk_hash", sa.CHAR(length=64), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.CheckConstraint("ordinal >= 0", name="ck_chunks_ordinal_nonnegative"),
        sa.ForeignKeyConstraint(
            ["document_version_id"], ["document_versions.document_version_id"]
        ),
        sa.PrimaryKeyConstraint("chunk_pk"),
        sa.UniqueConstraint(
            "document_version_id",
            "chunking_fingerprint",
            "compatibility_chunk_id",
            name="uq_chunks_version_fingerprint_compatibility_id",
        ),
        sa.UniqueConstraint(
            "document_version_id",
            "chunking_fingerprint",
            "ordinal",
            name="uq_chunks_version_fingerprint_ordinal",
        ),
    )
    op.create_table(
        "ingestion_run_documents",
        sa.Column("run_id", sa.Uuid(), nullable=False),
        sa.Column("document_pk", sa.Uuid(), nullable=False),
        sa.Column("change_type", sa.String(length=32), nullable=False),
        sa.Column("previous_version_id", sa.Uuid(), nullable=True),
        sa.Column("observed_version_id", sa.Uuid(), nullable=True),
        sa.Column("content_changed", sa.Boolean(), nullable=False),
        sa.Column("metadata_changed", sa.Boolean(), nullable=False),
        sa.ForeignKeyConstraint(["document_pk"], ["documents.document_pk"]),
        sa.ForeignKeyConstraint(
            ["observed_version_id"], ["document_versions.document_version_id"]
        ),
        sa.ForeignKeyConstraint(
            ["previous_version_id"], ["document_versions.document_version_id"]
        ),
        sa.ForeignKeyConstraint(["run_id"], ["ingestion_runs.run_id"]),
        sa.PrimaryKeyConstraint("run_id", "document_pk"),
    )
    op.create_table(
        "index_states",
        sa.Column("index_state_id", sa.Uuid(), nullable=False),
        sa.Column("corpus_id", sa.Uuid(), nullable=False),
        sa.Column("generation", sa.Integer(), nullable=False),
        sa.Column("ingestion_run_id", sa.Uuid(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("qdrant_collection", sa.String(length=255), nullable=False),
        sa.Column("snapshot_hash", sa.CHAR(length=64), nullable=False),
        sa.Column("bm25_corpus_hash", sa.CHAR(length=64), nullable=True),
        sa.Column("bm25_state_hash", sa.CHAR(length=64), nullable=True),
        sa.Column("embedding_fingerprint", sa.CHAR(length=64), nullable=False),
        sa.Column("embedding_dimension", sa.Integer(), nullable=False),
        sa.Column("chunking_fingerprint", sa.CHAR(length=64), nullable=False),
        sa.Column("document_count", sa.Integer(), nullable=False),
        sa.Column("chunk_count", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column("activated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("retired_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "chunk_count >= 0", name="ck_index_states_chunk_count_nonnegative"
        ),
        sa.CheckConstraint(
            "document_count >= 0", name="ck_index_states_document_count_nonnegative"
        ),
        sa.CheckConstraint(
            "embedding_dimension > 0", name="ck_index_states_embedding_dimension_positive"
        ),
        sa.CheckConstraint("generation > 0", name="ck_index_states_generation_positive"),
        sa.ForeignKeyConstraint(["corpus_id"], ["corpora.corpus_id"]),
        sa.ForeignKeyConstraint(["ingestion_run_id"], ["ingestion_runs.run_id"]),
        sa.PrimaryKeyConstraint("index_state_id"),
        sa.UniqueConstraint("corpus_id", "generation", name="uq_index_states_corpus_generation"),
        sa.UniqueConstraint("ingestion_run_id"),
        sa.UniqueConstraint("qdrant_collection"),
    )
    op.create_table(
        "index_state_chunks",
        sa.Column("index_state_id", sa.Uuid(), nullable=False),
        sa.Column("chunk_pk", sa.Uuid(), nullable=False),
        sa.Column("qdrant_point_id", sa.Uuid(), nullable=False),
        sa.Column("dense_action", sa.String(length=32), nullable=False),
        sa.Column("dense_source_chunk_pk", sa.Uuid(), nullable=True),
        sa.Column(
            "indexed_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["chunk_pk"], ["chunks.chunk_pk"]),
        sa.ForeignKeyConstraint(["dense_source_chunk_pk"], ["chunks.chunk_pk"]),
        sa.ForeignKeyConstraint(["index_state_id"], ["index_states.index_state_id"]),
        sa.PrimaryKeyConstraint("index_state_id", "chunk_pk"),
        sa.UniqueConstraint(
            "index_state_id", "qdrant_point_id", name="uq_index_state_chunks_state_point"
        ),
    )


def downgrade() -> None:
    """Drop only the application-lineage tables in reverse dependency order."""

    op.drop_table("index_state_chunks")
    op.drop_table("index_states")
    op.drop_table("ingestion_run_documents")
    op.drop_table("chunks")
    op.drop_table("document_versions")
    op.drop_table("documents")
    op.drop_table("ingestion_runs")
    op.drop_table("corpora")
