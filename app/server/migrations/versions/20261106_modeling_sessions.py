# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

"""Replace single-experiment fitting runs with dataset-level fitting sessions.

Existing fitting runs are intentionally discarded: they referenced a single
experiment and cannot be represented as dataset-level sessions. The generic
per-application parameter defaults are replaced by per-model global settings.

Revision ID: 20261106_modeling_sessions
Revises: 20261105_dataset_editing
"""

from alembic import op
import sqlalchemy as sa

from server.repositories.schemas import types as schema_types


revision = "20261106_modeling_sessions"
down_revision = "20261105_dataset_editing"
branch_labels = None
depends_on = None


METRIC_COLUMNS = (
    "sse",
    "rmse",
    "mae",
    "r_squared",
    "adjusted_r_squared",
    "chi_square",
    "aic",
    "aicc",
    "bic",
)


def _result_columns() -> list[sa.Column]:
    return [
        sa.Column("model_name", sa.String(length=128), nullable=False),
        sa.Column("model_version", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("convergence_message", sa.String(), nullable=False),
        sa.Column("function_evaluations", sa.Integer(), nullable=True),
        sa.Column("jacobian_rank", sa.Integer(), nullable=True),
        sa.Column("condition_number", sa.Float(), nullable=True),
        sa.Column("observation_count", sa.Integer(), nullable=False),
        sa.Column("parameter_count", sa.Integer(), nullable=False),
        *(sa.Column(name, sa.Float(), nullable=True) for name in METRIC_COLUMNS),
        sa.Column("predicted_observations", schema_types.JSONList(), nullable=False),
        sa.Column("predicted_curve", schema_types.JSONList(), nullable=False),
        sa.Column("warnings", schema_types.JSONList(), nullable=False),
        sa.Column("rank", sa.Integer(), nullable=True),
        sa.CheckConstraint(
            "status IN ('success', 'warning', 'failed')", name="ck_fit_results_status"
        ),
        sa.CheckConstraint(
            "observation_count >= 0", name="ck_fit_results_observations"
        ),
        sa.CheckConstraint("parameter_count > 0", name="ck_fit_results_parameters"),
        sa.PrimaryKeyConstraint("id"),
    ]


def _create_fit_parameters() -> None:
    op.create_table(
        "fit_parameters",
        sa.Column("result_id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("value_canonical", sa.Float(), nullable=False),
        sa.Column("standard_error_canonical", sa.Float(), nullable=True),
        sa.Column("ci95_low_canonical", sa.Float(), nullable=True),
        sa.Column("ci95_high_canonical", sa.Float(), nullable=True),
        sa.Column("unit_canonical", sa.String(length=128), nullable=False),
        sa.CheckConstraint("position >= 0", name="ck_fit_parameters_position"),
        sa.ForeignKeyConstraint(["result_id"], ["fit_results.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("result_id", "name"),
        sa.UniqueConstraint("result_id", "position", name="uq_fit_parameters_position"),
    )


def _drop_fit_results_and_parameters(index_name: str) -> None:
    op.drop_table("fit_parameters")
    op.drop_index(index_name, table_name="fit_results")
    op.drop_table("fit_results")


def upgrade() -> None:
    _drop_fit_results_and_parameters("ix_fit_results_run_rank")
    op.drop_index("ix_fitting_runs_isotherm_created", table_name="fitting_runs")
    op.drop_table("fitting_runs")

    op.create_table(
        "fitting_sessions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("dataset_id", sa.Integer(), nullable=False),
        sa.Column("dataset_name", sa.String(length=255), nullable=False),
        sa.Column("isotherm_ids", schema_types.JSONList(), nullable=False),
        sa.Column("input_sha256", sa.String(length=64), nullable=False),
        sa.Column("optimizer", sa.String(length=32), nullable=False),
        sa.Column("weighting", sa.String(length=32), nullable=False),
        sa.Column("max_evaluations", sa.Integer(), nullable=False),
        sa.Column("pressure_display_unit", sa.String(length=32), nullable=False),
        sa.Column("uptake_display_unit", sa.String(length=32), nullable=False),
        sa.Column("configuration", schema_types.JSONMapping(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("message", sa.String(), nullable=False),
        sa.Column("log", schema_types.JSONList(), nullable=False),
        sa.Column(
            "created_at", schema_types.UTCDateTime(timezone=True), nullable=False
        ),
        sa.Column(
            "completed_at", schema_types.UTCDateTime(timezone=True), nullable=True
        ),
        sa.CheckConstraint(
            "status IN ('running', 'completed', 'warning', 'failed', 'cancelled')",
            name="ck_fitting_sessions_status",
        ),
        sa.CheckConstraint(
            "max_evaluations > 0", name="ck_fitting_sessions_evaluations"
        ),
        sa.ForeignKeyConstraint(["dataset_id"], ["datasets.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_fitting_sessions_created", "fitting_sessions", ["created_at"])
    op.create_index("ix_fitting_sessions_dataset", "fitting_sessions", ["dataset_id"])

    op.create_table(
        "fitting_session_models",
        sa.Column("session_id", sa.Integer(), nullable=False),
        sa.Column("model_key", sa.String(length=64), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["session_id"], ["fitting_sessions.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("session_id", "model_key"),
    )
    op.create_index(
        "ix_fitting_session_models_key", "fitting_session_models", ["model_key"]
    )

    op.create_table(
        "fit_results",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("session_id", sa.Integer(), nullable=False),
        sa.Column("isotherm_id", sa.Integer(), nullable=True),
        sa.Column("isotherm_name", sa.String(length=255), nullable=False),
        *_result_columns(),
        sa.ForeignKeyConstraint(
            ["session_id"], ["fitting_sessions.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(["isotherm_id"], ["isotherms.id"], ondelete="SET NULL"),
        sa.UniqueConstraint(
            "session_id",
            "isotherm_id",
            "model_name",
            name="uq_fit_results_session_model",
        ),
    )
    op.create_index(
        "ix_fit_results_session_rank",
        "fit_results",
        ["session_id", "isotherm_id", "rank"],
    )
    _create_fit_parameters()

    op.create_table(
        "adsorption_model_settings",
        sa.Column("model_key", sa.String(length=64), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("parameter_bounds", schema_types.JSONMapping(), nullable=False),
        sa.Column(
            "updated_at", schema_types.UTCDateTime(timezone=True), nullable=False
        ),
        sa.PrimaryKeyConstraint("model_key"),
    )

    with op.batch_alter_table("application_settings") as batch:
        batch.drop_column("fitting_default_parameter_initial")
        batch.drop_column("fitting_default_parameter_min")
        batch.drop_column("fitting_default_parameter_max")


def downgrade() -> None:
    with op.batch_alter_table("application_settings") as batch:
        batch.add_column(
            sa.Column(
                "fitting_default_parameter_initial",
                sa.Float(),
                nullable=False,
                server_default="1.0",
            )
        )
        batch.add_column(
            sa.Column(
                "fitting_default_parameter_min",
                sa.Float(),
                nullable=False,
                server_default="0.0",
            )
        )
        batch.add_column(
            sa.Column(
                "fitting_default_parameter_max",
                sa.Float(),
                nullable=False,
                server_default="100.0",
            )
        )
    op.drop_table("adsorption_model_settings")
    _drop_fit_results_and_parameters("ix_fit_results_session_rank")
    op.drop_index("ix_fitting_session_models_key", table_name="fitting_session_models")
    op.drop_table("fitting_session_models")
    op.drop_index("ix_fitting_sessions_dataset", table_name="fitting_sessions")
    op.drop_index("ix_fitting_sessions_created", table_name="fitting_sessions")
    op.drop_table("fitting_sessions")

    op.create_table(
        "fitting_runs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("isotherm_id", sa.Integer(), nullable=False),
        sa.Column("input_sha256", sa.String(length=64), nullable=False),
        sa.Column("optimizer", sa.String(length=32), nullable=False),
        sa.Column("max_evaluations", sa.Integer(), nullable=False),
        sa.Column("pressure_display_unit", sa.String(length=32), nullable=False),
        sa.Column("uptake_display_unit", sa.String(length=32), nullable=False),
        sa.Column("configuration", schema_types.JSONMapping(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("message", sa.String(), nullable=False),
        sa.Column(
            "created_at", schema_types.UTCDateTime(timezone=True), nullable=False
        ),
        sa.Column(
            "completed_at", schema_types.UTCDateTime(timezone=True), nullable=True
        ),
        sa.CheckConstraint(
            "status IN ('running', 'completed', 'warning', 'failed', 'cancelled')",
            name="ck_fitting_runs_status",
        ),
        sa.CheckConstraint("max_evaluations > 0", name="ck_fitting_runs_evaluations"),
        sa.ForeignKeyConstraint(["isotherm_id"], ["isotherms.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_fitting_runs_isotherm_created",
        "fitting_runs",
        ["isotherm_id", "created_at"],
    )
    op.create_table(
        "fit_results",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("run_id", sa.Integer(), nullable=False),
        *_result_columns(),
        sa.ForeignKeyConstraint(["run_id"], ["fitting_runs.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("run_id", "model_name", name="uq_fit_results_run_model"),
    )
    op.create_index("ix_fit_results_run_rank", "fit_results", ["run_id", "rank"])
    _create_fit_parameters()
