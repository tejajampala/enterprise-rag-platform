"""Add immutable derived PII classification metadata.

Revision ID: 0004_pii_metadata
Revises: 0003_access_policy
Create Date: 2026-09-21
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0004_pii_metadata"
down_revision: str | None = "0003_access_policy"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

JSON_DOCUMENT_TYPE = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    """Create the derived classification table without changing source versions."""

    op.create_table(
        "document_version_pii",
        sa.Column("document_version_id", sa.Uuid(), nullable=False),
        sa.Column("detector_version", sa.String(length=255), nullable=False),
        sa.Column("classification_hash", sa.CHAR(length=64), nullable=False),
        sa.Column(
            "pii_types",
            JSON_DOCUMENT_TYPE,
            server_default=sa.text("'[]'"),
            nullable=False,
        ),
        sa.Column("match_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column(
            "classified_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "match_count >= 0",
            name="ck_document_version_pii_match_count_nonnegative",
        ),
        sa.ForeignKeyConstraint(
            ["document_version_id"],
            ["document_versions.document_version_id"],
        ),
        sa.PrimaryKeyConstraint("document_version_id", "detector_version"),
    )


def downgrade() -> None:
    """Remove only derived PII classification metadata."""

    op.drop_table("document_version_pii")
