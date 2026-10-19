# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

"""Persist public-source health and acquisition runs.

Revision ID: 20261019_public_data_sync
Revises: 20260924_fitting_cancel
"""

from __future__ import annotations

from typing import Sequence

from alembic import op
import sqlalchemy as sa

from server.repositories.schemas import types as schema_types


revision: str = "20261019_public_data_sync"
down_revision: str | None = "20260924_fitting_cancel"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

###############################################################################
def upgrade() -> None:
    # Additive ALTER TABLE statements keep existing source-record foreign keys
    # intact on SQLite, where batch alteration would rebuild and drop the table.
    op.execute(
        "ALTER TABLE data_sources ADD COLUMN health_status VARCHAR(16) "
        "NOT NULL DEFAULT 'unknown' "
        "CONSTRAINT ck_data_sources_health_status CHECK "
        "(health_status IN ('available', 'degraded', 'unavailable', 'unknown'))"
    )
    op.add_column(
        "data_sources", sa.Column("health_detail", sa.String(length=2000))
    )
    op.add_column(
        "data_sources",
        sa.Column("last_health_checked_at", schema_types.UTCDateTime(timezone=True)),
    )
    op.add_column(
        "data_sources",
        sa.Column("last_successful_sync_at", schema_types.UTCDateTime(timezone=True)),
    )

    op.create_table(
        "public_data_sync_runs",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column(
            "source_key",
            sa.String(length=64),
            sa.ForeignKey("data_sources.key", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("operation", sa.String(length=32), nullable=False),
        sa.Column("parameters", schema_types.JSONMapping(), nullable=False),
        sa.Column("status", sa.String(length=24), nullable=False, server_default="queued"),
        sa.Column("progress", sa.Float(), nullable=False, server_default="0"),
        sa.Column("processed_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("imported_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("skipped_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("failed_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("checkpoint", schema_types.JSONMapping(), nullable=False),
        sa.Column("cancel_requested", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("error_detail", sa.String(length=2000)),
        sa.Column(
            "created_at",
            schema_types.UTCDateTime(timezone=True),
            nullable=False,
        ),
        sa.Column("started_at", schema_types.UTCDateTime(timezone=True)),
        sa.Column("finished_at", schema_types.UTCDateTime(timezone=True)),
        sa.Column(
            "updated_at",
            schema_types.UTCDateTime(timezone=True),
            nullable=False,
        ),
        sa.CheckConstraint(
            "status IN ('queued', 'running', 'completed', 'partially_completed', "
            "'failed', 'cancelled', 'interrupted')",
            name="ck_public_data_sync_runs_status",
        ),
        sa.CheckConstraint(
            "operation IN ('health_check', 'nist_category', 'pubchem_resolve', 'cod_import')",
            name="ck_public_data_sync_runs_operation",
        ),
        sa.CheckConstraint(
            "progress >= 0 AND progress <= 100",
            name="ck_public_data_sync_runs_progress",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_public_data_sync_runs_queue",
        "public_data_sync_runs",
        ["status", "created_at"],
    )
    op.create_index(
        "ix_public_data_sync_runs_source_created",
        "public_data_sync_runs",
        ["source_key", "created_at"],
    )
    op.create_table(
        "public_data_sync_failures",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column(
            "sync_run_id",
            sa.String(length=36),
            sa.ForeignKey("public_data_sync_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "source_key",
            sa.String(length=64),
            sa.ForeignKey("data_sources.key", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("external_id", sa.String(length=512)),
        sa.Column("reason", sa.String(length=2000), nullable=False),
        sa.Column("retry_eligible_at", schema_types.UTCDateTime(timezone=True)),
        sa.Column(
            "created_at",
            schema_types.UTCDateTime(timezone=True),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_public_data_sync_failures_run",
        "public_data_sync_failures",
        ["sync_run_id", "id"],
    )
    op.create_index(
        "ix_public_data_sync_failures_retry",
        "public_data_sync_failures",
        ["source_key", "retry_eligible_at"],
    )

###############################################################################
def downgrade() -> None:
    op.drop_index(
        "ix_public_data_sync_failures_retry",
        table_name="public_data_sync_failures",
    )
    op.drop_index(
        "ix_public_data_sync_failures_run",
        table_name="public_data_sync_failures",
    )
    op.drop_table("public_data_sync_failures")
    op.drop_index(
        "ix_public_data_sync_runs_source_created",
        table_name="public_data_sync_runs",
    )
    op.drop_index(
        "ix_public_data_sync_runs_queue",
        table_name="public_data_sync_runs",
    )
    op.drop_table("public_data_sync_runs")
    op.drop_column("data_sources", "last_successful_sync_at")
    op.drop_column("data_sources", "last_health_checked_at")
    op.drop_column("data_sources", "health_detail")
    op.drop_column("data_sources", "health_status")
