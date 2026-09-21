"""Persist access policy with immutable document versions.

Revision ID: 0003_access_policy
Revises: 0002_managed_index_generations
Create Date: 2026-09-21
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0003_access_policy"
down_revision: str | None = "0002_managed_index_generations"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

JSON_DOCUMENT_TYPE = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    """Add safe PUBLIC defaults for all existing immutable versions."""

    with op.batch_alter_table("document_versions") as batch_op:
        batch_op.add_column(
            sa.Column(
                "access_visibility",
                sa.String(length=32),
                nullable=False,
                server_default="PUBLIC",
            )
        )
        batch_op.add_column(
            sa.Column(
                "allowed_groups",
                JSON_DOCUMENT_TYPE,
                nullable=False,
                server_default=sa.text("'[]'"),
            )
        )
        batch_op.create_check_constraint(
            "ck_document_versions_access_visibility",
            "access_visibility IN ('PUBLIC', 'RESTRICTED')",
        )


def downgrade() -> None:
    """Remove document-version access policy fields."""

    with op.batch_alter_table("document_versions") as batch_op:
        batch_op.drop_constraint(
            "ck_document_versions_access_visibility",
            type_="check",
        )
        batch_op.drop_column("allowed_groups")
        batch_op.drop_column("access_visibility")
