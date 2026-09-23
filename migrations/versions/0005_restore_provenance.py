"""Add managed-generation restoration provenance.

Revision ID: 0005_restore_provenance
Revises: 0004_pii_metadata
Create Date: 2026-09-22
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005_restore_provenance"
down_revision: str | None = "0004_pii_metadata"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add nullable, auditable self-reference fields for restored generations."""

    with op.batch_alter_table("index_states") as batch_op:
        batch_op.add_column(
            sa.Column("restored_from_index_state_id", sa.Uuid(), nullable=True)
        )
        batch_op.add_column(sa.Column("restoration_reason", sa.Text(), nullable=True))
        batch_op.create_foreign_key(
            "fk_index_states_restored_from",
            "index_states",
            ["restored_from_index_state_id"],
            ["index_state_id"],
        )


def downgrade() -> None:
    """Remove restoration provenance without rewriting generation history."""

    with op.batch_alter_table("index_states") as batch_op:
        batch_op.drop_constraint("fk_index_states_restored_from", type_="foreignkey")
        batch_op.drop_column("restoration_reason")
        batch_op.drop_column("restored_from_index_state_id")
