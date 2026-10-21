# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

"""Move application settings into the active operational database.

Revision ID: 20261020_app_settings
Revises: 20261020_public_data_nav
"""

from __future__ import annotations

from typing import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "20261020_app_settings"
down_revision: str | None = "20261020_public_data_nav"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


###############################################################################
def upgrade() -> None:
    bind = op.get_bind()
    if "application_settings" in sa.inspect(bind).get_table_names():
        return
    op.create_table(
        "application_settings",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("runtime_host", sa.String(16), nullable=False),
        sa.Column("runtime_backend_port", sa.Integer(), nullable=False),
        sa.Column("runtime_frontend_port", sa.Integer(), nullable=False),
        sa.Column("datasets_allowed_extensions", sa.JSON(), nullable=False),
        sa.Column("datasets_column_detection_cutoff", sa.Float(), nullable=False),
        sa.Column("nist_parallel_tasks", sa.Integer(), nullable=False),
        sa.Column("public_data_request_timeout_seconds", sa.Float(), nullable=False),
        sa.Column("public_data_retry_attempts", sa.Integer(), nullable=False),
        sa.Column("public_data_retry_backoff_max_seconds", sa.Float(), nullable=False),
        sa.Column("public_data_pubchem_parallel_requests", sa.Integer(), nullable=False),
        sa.Column("public_data_cod_max_interactive_results", sa.Integer(), nullable=False),
        sa.Column("public_data_health_cache_ttl_seconds", sa.Integer(), nullable=False),
        sa.Column("public_data_sync_batch_size", sa.Integer(), nullable=False),
        sa.Column("public_data_sync_max_records_per_run", sa.Integer(), nullable=False),
        sa.Column("public_data_sync_max_concurrent_jobs", sa.Integer(), nullable=False),
        sa.Column("public_data_sync_response_size_limit_bytes", sa.Integer(), nullable=False),
        sa.Column("public_data_sync_resume_on_startup", sa.Boolean(), nullable=False),
        sa.Column("public_data_auto_sync_enabled", sa.Boolean(), nullable=False),
        sa.Column("public_data_auto_sync_categories", sa.JSON(), nullable=False),
        sa.Column("public_data_auto_sync_interval_hours", sa.Integer(), nullable=False),
        sa.Column("public_data_auto_sync_fraction", sa.Float(), nullable=False),
        sa.Column("fitting_default_max_iterations", sa.Integer(), nullable=False),
        sa.Column("fitting_max_iterations_upper_bound", sa.Integer(), nullable=False),
        sa.Column("fitting_default_parameter_initial", sa.Float(), nullable=False),
        sa.Column("fitting_default_parameter_min", sa.Float(), nullable=False),
        sa.Column("fitting_default_parameter_max", sa.Float(), nullable=False),
        sa.Column("fitting_preview_row_limit", sa.Integer(), nullable=False),
        sa.Column("fitting_best_model_metric", sa.String(32), nullable=False),
        sa.Column("jobs_polling_interval", sa.Float(), nullable=False),
        sa.Column("training_use_jit", sa.Boolean(), nullable=False),
        sa.Column("training_jit_backend", sa.String(32), nullable=False),
        sa.Column("training_use_mixed_precision", sa.Boolean(), nullable=False),
        sa.Column("training_dataloader_workers", sa.Integer(), nullable=False),
        sa.Column("training_persistent_workers", sa.Boolean(), nullable=False),
        sa.Column("updated_at", sa.String(40), nullable=False),
    )


###############################################################################
def downgrade() -> None:
    # Settings rows may predate this migration and are user data.
    pass
