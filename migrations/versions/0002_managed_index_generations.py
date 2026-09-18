"""Add managed-index generation metadata and state constraints.

Revision ID: 0002_managed_index_generations
Revises: 0001_application_lineage
Create Date: 2026-09-18
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0002_managed_index_generations"
down_revision: str | None = "0001_application_lineage"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

NAMING_CONVENTION = {"uq": "uq_%(table_name)s_%(column_0_name)s"}
JSON_DOCUMENT_TYPE = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def _ingestion_run_unique_constraint_name() -> str:
    constraints = sa.inspect(op.get_bind()).get_unique_constraints("index_states")
    for constraint in constraints:
        if constraint.get("column_names") == ["ingestion_run_id"]:
            name = constraint.get("name")
            if isinstance(name, str):
                return name
            return "uq_index_states_ingestion_run_id"
    raise RuntimeError("index_states ingestion_run_id unique constraint was not found")


def upgrade() -> None:
    """Store complete managed-generation metadata and enforce safe identities."""

    ingestion_run_unique = _ingestion_run_unique_constraint_name()
    with op.batch_alter_table(
        "index_states", naming_convention=NAMING_CONVENTION
    ) as batch_op:
        batch_op.drop_constraint(ingestion_run_unique, type_="unique")
        batch_op.drop_constraint(
            "ck_index_states_embedding_dimension_positive", type_="check"
        )
        batch_op.alter_column(
            "embedding_dimension",
            existing_type=sa.Integer(),
            existing_nullable=False,
            nullable=True,
        )
        batch_op.add_column(sa.Column("bm25_state", JSON_DOCUMENT_TYPE, nullable=True))
        batch_op.add_column(
            sa.Column("index_configuration", JSON_DOCUMENT_TYPE, nullable=True)
        )
        batch_op.add_column(
            sa.Column("index_configuration_fingerprint", sa.CHAR(length=64), nullable=True)
        )
        batch_op.add_column(sa.Column("error_message", sa.Text(), nullable=True))
        batch_op.add_column(
            sa.Column("failed_at", sa.DateTime(timezone=True), nullable=True)
        )
        batch_op.create_check_constraint(
            "ck_index_states_embedding_dimension_positive",
            "embedding_dimension IS NULL OR embedding_dimension > 0",
        )

    with op.batch_alter_table("index_state_chunks") as batch_op:
        batch_op.create_check_constraint(
            "ck_index_state_chunks_dense_provenance",
            "(dense_action = 'COMPUTED' AND dense_source_chunk_pk IS NULL) OR "
            "(dense_action = 'REUSED' AND dense_source_chunk_pk IS NOT NULL)",
        )

    op.create_index(
        "uq_index_states_one_active_per_corpus",
        "index_states",
        ["corpus_id"],
        unique=True,
        postgresql_where=sa.text("status = 'ACTIVE'"),
        sqlite_where=sa.text("status = 'ACTIVE'"),
    )


def downgrade() -> None:
    """Restore the initial one-index-attempt-per-ingestion-run schema."""

    op.drop_index("uq_index_states_one_active_per_corpus", table_name="index_states")

    with op.batch_alter_table("index_state_chunks") as batch_op:
        batch_op.drop_constraint(
            "ck_index_state_chunks_dense_provenance", type_="check"
        )

    with op.batch_alter_table(
        "index_states", naming_convention=NAMING_CONVENTION
    ) as batch_op:
        batch_op.drop_constraint(
            "ck_index_states_embedding_dimension_positive", type_="check"
        )
        batch_op.drop_column("failed_at")
        batch_op.drop_column("error_message")
        batch_op.drop_column("index_configuration_fingerprint")
        batch_op.drop_column("index_configuration")
        batch_op.drop_column("bm25_state")
        batch_op.alter_column(
            "embedding_dimension",
            existing_type=sa.Integer(),
            existing_nullable=True,
            nullable=False,
        )
        batch_op.create_check_constraint(
            "ck_index_states_embedding_dimension_positive",
            "embedding_dimension > 0",
        )
        batch_op.create_unique_constraint(
            "uq_index_states_ingestion_run_id", ["ingestion_run_id"]
        )
