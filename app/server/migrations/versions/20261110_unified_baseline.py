# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

"""Create the unified ADSMOD schema for a fresh database.

Adsorption data is one long-format measurement table: an experiment
(`isotherms`) has one adsorbent and 1..N species (`isotherm_components`), and
each `observations` row is the uptake of one species at one canonical pressure,
unique on (isotherm_id, adsorbate_id, pressure_key). Databases stamped with any
earlier revision must be recreated; startup refuses to infer their schema.
"""

from alembic import op
import sqlalchemy as sa

from server.repositories.schemas import types as schema_types


revision = "20261110_unified_baseline"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:  # noqa: PLR0915
    op.create_table(
        "adsorbates",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("key", sa.String(length=255), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("normalized_name", sa.String(length=255), nullable=False),
        sa.Column("inchi_key", sa.String(length=27), nullable=True),
        sa.Column("inchi", sa.String(), nullable=True),
        sa.Column("formula", sa.String(length=255), nullable=True),
        sa.Column("molar_mass_g_mol", sa.Float(), nullable=True),
        sa.Column("smiles", sa.String(), nullable=True),
        sa.Column(
            "identity_status",
            sa.String(length=16),
            server_default="provisional",
            nullable=False,
        ),
        sa.Column(
            "created_at", schema_types.UTCDateTime(timezone=True), nullable=False
        ),
        sa.Column(
            "updated_at", schema_types.UTCDateTime(timezone=True), nullable=False
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("inchi_key"),
        sa.UniqueConstraint("key"),
    )
    op.create_index("ix_adsorbates_formula", "adsorbates", ["formula"], unique=False)
    op.create_index(
        "ix_adsorbates_molar_mass", "adsorbates", ["molar_mass_g_mol"], unique=False
    )
    op.create_index(
        "ix_adsorbates_normalized_name", "adsorbates", ["normalized_name"], unique=False
    )
    op.create_table(
        "adsorbents",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("key", sa.String(length=255), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("normalized_name", sa.String(length=255), nullable=False),
        sa.Column("external_identifier", sa.String(length=255), nullable=True),
        sa.Column("formula", sa.String(length=255), nullable=True),
        sa.Column("molar_mass_g_mol", sa.Float(), nullable=True),
        sa.Column("smiles", sa.String(), nullable=True),
        sa.Column(
            "identity_status",
            sa.String(length=16),
            server_default="provisional",
            nullable=False,
        ),
        sa.Column("scope_dataset_id", sa.Integer(), nullable=True),
        sa.Column(
            "created_at", schema_types.UTCDateTime(timezone=True), nullable=False
        ),
        sa.Column(
            "updated_at", schema_types.UTCDateTime(timezone=True), nullable=False
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("external_identifier"),
        sa.UniqueConstraint("key"),
    )
    op.create_index("ix_adsorbents_formula", "adsorbents", ["formula"], unique=False)
    op.create_index(
        "ix_adsorbents_normalized_name", "adsorbents", ["normalized_name"], unique=False
    )
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
    op.create_table(
        "application_settings",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("datasets_allowed_extensions", sa.JSON(), nullable=False),
        sa.Column("datasets_column_detection_cutoff", sa.Float(), nullable=False),
        sa.Column("nist_parallel_tasks", sa.Integer(), nullable=False),
        sa.Column("public_data_request_timeout_seconds", sa.Float(), nullable=False),
        sa.Column("public_data_retry_attempts", sa.Integer(), nullable=False),
        sa.Column("public_data_retry_backoff_max_seconds", sa.Float(), nullable=False),
        sa.Column(
            "public_data_pubchem_parallel_requests", sa.Integer(), nullable=False
        ),
        sa.Column(
            "public_data_cod_max_interactive_results", sa.Integer(), nullable=False
        ),
        sa.Column("public_data_health_cache_ttl_seconds", sa.Integer(), nullable=False),
        sa.Column("public_data_sync_batch_size", sa.Integer(), nullable=False),
        sa.Column("public_data_sync_max_records_per_run", sa.Integer(), nullable=False),
        sa.Column("public_data_sync_max_concurrent_jobs", sa.Integer(), nullable=False),
        sa.Column(
            "public_data_auto_resolve_uploaded_species",
            sa.Boolean(),
            server_default=sa.text("1"),
            nullable=False,
        ),
        sa.Column(
            "public_data_sync_response_size_limit_bytes", sa.Integer(), nullable=False
        ),
        sa.Column("public_data_sync_resume_on_startup", sa.Boolean(), nullable=False),
        sa.Column("public_data_auto_sync_enabled", sa.Boolean(), nullable=False),
        sa.Column("public_data_auto_sync_categories", sa.JSON(), nullable=False),
        sa.Column("public_data_auto_sync_interval_hours", sa.Integer(), nullable=False),
        sa.Column("public_data_auto_sync_fraction", sa.Float(), nullable=False),
        sa.Column("fitting_default_max_iterations", sa.Integer(), nullable=False),
        sa.Column("fitting_max_iterations_upper_bound", sa.Integer(), nullable=False),
        sa.Column("fitting_preview_row_limit", sa.Integer(), nullable=False),
        sa.Column("fitting_best_model_metric", sa.String(length=32), nullable=False),
        sa.Column("jobs_polling_interval", sa.Float(), nullable=False),
        sa.Column("training_use_jit", sa.Boolean(), nullable=False),
        sa.Column("training_jit_backend", sa.String(length=32), nullable=False),
        sa.Column("training_use_mixed_precision", sa.Boolean(), nullable=False),
        sa.Column("training_dataloader_workers", sa.Integer(), nullable=False),
        sa.Column("training_persistent_workers", sa.Boolean(), nullable=False),
        sa.Column("updated_at", sa.String(length=40), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "custom_units",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("quantity", sa.String(length=16), nullable=False),
        sa.Column("symbol", sa.String(length=64), nullable=False),
        sa.Column("normalized_symbol", sa.String(length=64), nullable=False),
        sa.Column("factor_to_canonical", sa.Float(), nullable=True),
        sa.Column(
            "created_at", schema_types.UTCDateTime(timezone=True), nullable=False
        ),
        sa.CheckConstraint(
            "quantity IN ('pressure', 'uptake', 'temperature')",
            name="ck_custom_units_quantity",
        ),
        sa.CheckConstraint(
            "factor_to_canonical IS NULL OR factor_to_canonical > 0",
            name="ck_custom_units_factor",
        ),
        sa.CheckConstraint(
            "length(normalized_symbol) > 0", name="ck_custom_units_symbol"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "quantity", "normalized_symbol", name="uq_custom_units_quantity_symbol"
        ),
    )
    op.create_table(
        "data_sources",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("key", sa.String(length=64), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("description", sa.String(), nullable=False),
        sa.Column("homepage_url", sa.String(length=1024), nullable=False),
        sa.Column("license_name", sa.String(length=128), nullable=True),
        sa.Column("license_url", sa.String(length=1024), nullable=True),
        sa.Column("terms_url", sa.String(length=1024), nullable=True),
        sa.Column("capabilities", schema_types.JSONList(), nullable=False),
        sa.Column(
            "health_status",
            sa.String(length=16),
            server_default="unknown",
            nullable=False,
        ),
        sa.Column("health_detail", sa.String(length=2000), nullable=True),
        sa.Column(
            "last_health_checked_at",
            schema_types.UTCDateTime(timezone=True),
            nullable=True,
        ),
        sa.Column(
            "last_successful_sync_at",
            schema_types.UTCDateTime(timezone=True),
            nullable=True,
        ),
        sa.Column(
            "created_at", schema_types.UTCDateTime(timezone=True), nullable=False
        ),
        sa.Column(
            "updated_at", schema_types.UTCDateTime(timezone=True), nullable=False
        ),
        sa.CheckConstraint(
            "health_status IN ('available', 'degraded', 'unavailable', 'unknown')",
            name="ck_data_sources_health_status",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("key"),
    )
    op.create_table(
        "datasets",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("normalized_name", sa.String(length=255), nullable=False),
        sa.Column("source", sa.String(length=16), nullable=False),
        sa.Column("description", sa.String(), nullable=False),
        sa.Column("tags", schema_types.JSONList(), nullable=False),
        sa.Column("provenance", schema_types.JSONMapping(), nullable=False),
        sa.Column(
            "created_at", schema_types.UTCDateTime(timezone=True), nullable=False
        ),
        sa.Column(
            "updated_at", schema_types.UTCDateTime(timezone=True), nullable=False
        ),
        sa.CheckConstraint("source IN ('uploaded', 'nist')", name="ck_datasets_source"),
        sa.CheckConstraint("length(normalized_name) > 0", name="ck_datasets_name"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("normalized_name", name="uq_datasets_normalized_name"),
    )
    op.create_index(
        "ix_datasets_source_created_at",
        "datasets",
        ["source", "created_at"],
        unique=False,
    )
    op.create_table(
        "identity_reviews",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("entity_kind", sa.String(length=16), nullable=False),
        sa.Column("entity_id", sa.Integer(), nullable=True),
        sa.Column("candidate_id", sa.Integer(), nullable=True),
        sa.Column("raw_name", sa.String(length=1024), nullable=False),
        sa.Column("reason", sa.String(length=32), nullable=False),
        sa.Column("score", sa.Float(), nullable=True),
        sa.Column("evidence", schema_types.JSONMapping(), nullable=False),
        sa.Column("fingerprint", sa.String(length=128), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("merge_snapshot", schema_types.JSONMapping(), nullable=True),
        sa.Column(
            "created_at", schema_types.UTCDateTime(timezone=True), nullable=False
        ),
        sa.Column("decided_at", schema_types.UTCDateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "entity_kind IN ('adsorbate', 'adsorbent')", name="ck_identity_reviews_kind"
        ),
        sa.CheckConstraint(
            "status IN ('open', 'merged', 'rejected')",
            name="ck_identity_reviews_status",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("fingerprint"),
    )
    op.create_index(
        "ix_identity_reviews_status",
        "identity_reviews",
        ["status", "entity_kind"],
        unique=False,
    )
    op.create_table(
        "references",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("doi", sa.String(length=255), nullable=True),
        sa.Column("title", sa.String(length=2048), nullable=True),
        sa.Column("journal", sa.String(length=512), nullable=True),
        sa.Column("year", sa.Integer(), nullable=True),
        sa.Column("url", sa.String(length=2048), nullable=True),
        sa.Column(
            "created_at", schema_types.UTCDateTime(timezone=True), nullable=False
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("doi"),
    )
    op.create_index("ix_references_year", "references", ["year"], unique=False)
    op.create_table(
        "training_snapshots",
        sa.Column("snapshot_id", sa.String(length=36), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column(
            "created_at", schema_types.UTCDateTime(timezone=True), nullable=False
        ),
        sa.Column("row_count", sa.Integer(), nullable=False),
        sa.Column("metadata", schema_types.JSONMapping(), nullable=False),
        sa.CheckConstraint("row_count > 0", name="ck_training_snapshots_row_count"),
        sa.PrimaryKeyConstraint("snapshot_id"),
        sa.UniqueConstraint("content_hash"),
    )
    op.create_table(
        "dataset_imports",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("dataset_id", sa.Integer(), nullable=False),
        sa.Column("original_filename", sa.String(length=512), nullable=False),
        sa.Column("source_sha256", sa.String(length=64), nullable=False),
        sa.Column("mapping_sha256", sa.String(length=64), nullable=False),
        sa.Column("source_structure", sa.String(length=16), nullable=False),
        sa.Column("parser_version", sa.String(length=32), nullable=False),
        sa.Column("column_mapping", schema_types.JSONMapping(), nullable=False),
        sa.Column("validation_result", schema_types.JSONMapping(), nullable=False),
        sa.Column("warnings", schema_types.JSONList(), nullable=False),
        sa.Column(
            "created_at", schema_types.UTCDateTime(timezone=True), nullable=False
        ),
        sa.CheckConstraint(
            "source_structure IN ('atomic', 'aggregated', 'mixed')",
            name="ck_dataset_import_structure",
        ),
        sa.ForeignKeyConstraint(["dataset_id"], ["datasets.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "dataset_id", "source_sha256", name="uq_dataset_import_source"
        ),
    )
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
    op.create_index(
        "ix_fitting_sessions_created", "fitting_sessions", ["created_at"], unique=False
    )
    op.create_index(
        "ix_fitting_sessions_dataset", "fitting_sessions", ["dataset_id"], unique=False
    )
    op.create_table(
        "isotherms",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("dataset_id", sa.Integer(), nullable=False),
        sa.Column("external_key", sa.String(length=255), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("adsorbent_id", sa.Integer(), nullable=False),
        sa.Column("adsorbent_source_name", sa.String(length=255), nullable=True),
        sa.Column("temperature_original", sa.Float(), nullable=False),
        sa.Column("temperature_original_unit", sa.String(length=16), nullable=False),
        sa.Column("temperature_k", sa.Float(), nullable=False),
        sa.Column("pressure_basis", sa.String(length=16), nullable=False),
        sa.Column("saturation_pressure_pa", sa.Float(), nullable=True),
        sa.Column("conditions", schema_types.JSONMapping(), nullable=False),
        sa.Column("provenance", schema_types.JSONMapping(), nullable=False),
        sa.Column(
            "created_at", schema_types.UTCDateTime(timezone=True), nullable=False
        ),
        sa.Column(
            "updated_at", schema_types.UTCDateTime(timezone=True), nullable=False
        ),
        sa.Column("edited_at", schema_types.UTCDateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "pressure_basis IN ('absolute', 'partial', 'relative')",
            name="ck_isotherms_pressure_basis",
        ),
        sa.CheckConstraint(
            "saturation_pressure_pa IS NULL OR saturation_pressure_pa > 0",
            name="ck_isotherms_saturation_pressure",
        ),
        sa.CheckConstraint("temperature_k > 0", name="ck_isotherms_temperature"),
        sa.ForeignKeyConstraint(
            ["adsorbent_id"], ["adsorbents.id"], ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(["dataset_id"], ["datasets.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "dataset_id", "external_key", name="uq_isotherms_dataset_key"
        ),
    )
    op.create_index(
        "ix_isotherms_adsorbent", "isotherms", ["adsorbent_id"], unique=False
    )
    op.create_index(
        "ix_isotherms_dataset_edited",
        "isotherms",
        ["dataset_id", "edited_at"],
        unique=False,
    )
    op.create_index(
        "ix_isotherms_dataset_name", "isotherms", ["dataset_id", "name"], unique=False
    )
    op.create_index(
        "ix_isotherms_temperature", "isotherms", ["temperature_k"], unique=False
    )
    op.create_table(
        "public_data_sync_runs",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("source_key", sa.String(length=64), nullable=False),
        sa.Column("operation", sa.String(length=32), nullable=False),
        sa.Column("parameters", schema_types.JSONMapping(), nullable=False),
        sa.Column(
            "status", sa.String(length=24), server_default="queued", nullable=False
        ),
        sa.Column("progress", sa.Float(), server_default="0", nullable=False),
        sa.Column("processed_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("imported_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("skipped_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("failed_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("checkpoint", schema_types.JSONMapping(), nullable=False),
        sa.Column(
            "cancel_requested",
            sa.Boolean(),
            server_default=sa.text("0"),
            nullable=False,
        ),
        sa.Column("error_detail", sa.String(length=2000), nullable=True),
        sa.Column(
            "created_at", schema_types.UTCDateTime(timezone=True), nullable=False
        ),
        sa.Column("started_at", schema_types.UTCDateTime(timezone=True), nullable=True),
        sa.Column(
            "finished_at", schema_types.UTCDateTime(timezone=True), nullable=True
        ),
        sa.Column(
            "updated_at", schema_types.UTCDateTime(timezone=True), nullable=False
        ),
        sa.CheckConstraint(
            "operation IN ('health_check', 'nist_index', 'nist_category', 'pubchem_resolve', 'pubchem_enrich', 'cod_import')",
            name="ck_public_data_sync_runs_operation",
        ),
        sa.CheckConstraint(
            "status IN ('queued', 'running', 'completed', 'partially_completed', 'failed', 'cancelled', 'interrupted')",
            name="ck_public_data_sync_runs_status",
        ),
        sa.CheckConstraint(
            "progress >= 0 AND progress <= 100",
            name="ck_public_data_sync_runs_progress",
        ),
        sa.ForeignKeyConstraint(
            ["source_key"], ["data_sources.key"], ondelete="RESTRICT"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_public_data_sync_runs_queue",
        "public_data_sync_runs",
        ["status", "created_at"],
        unique=False,
    )
    op.create_index(
        "ix_public_data_sync_runs_source_created",
        "public_data_sync_runs",
        ["source_key", "created_at"],
        unique=False,
    )
    op.create_table(
        "source_records",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("source_id", sa.Integer(), nullable=False),
        sa.Column("record_type", sa.String(length=32), nullable=False),
        sa.Column("external_id", sa.String(length=512), nullable=False),
        sa.Column("source_url", sa.String(length=2048), nullable=True),
        sa.Column("source_version", sa.String(length=128), nullable=True),
        sa.Column(
            "retrieved_at", schema_types.UTCDateTime(timezone=True), nullable=False
        ),
        sa.Column("raw_metadata", schema_types.JSONMapping(), nullable=False),
        sa.CheckConstraint(
            "record_type IN ('adsorption', 'chemical', 'material', 'structure')",
            name="ck_source_records_type",
        ),
        sa.ForeignKeyConstraint(
            ["source_id"], ["data_sources.id"], ondelete="RESTRICT"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "source_id", "record_type", "external_id", name="uq_source_records_identity"
        ),
    )
    op.create_index(
        "ix_source_records_source_retrieved",
        "source_records",
        ["source_id", "retrieved_at"],
        unique=False,
    )
    op.create_table(
        "structures",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("adsorbent_id", sa.Integer(), nullable=True),
        sa.Column("name", sa.String(length=512), nullable=True),
        sa.Column("formula", sa.String(length=512), nullable=True),
        sa.Column("format", sa.String(length=16), nullable=False),
        sa.Column("content", sa.String(), nullable=False),
        sa.Column("content_sha256", sa.String(length=64), nullable=False),
        sa.Column("space_group", sa.String(length=128), nullable=True),
        sa.Column("space_group_number", sa.Integer(), nullable=True),
        sa.Column("cell_a_angstrom", sa.Float(), nullable=True),
        sa.Column("cell_b_angstrom", sa.Float(), nullable=True),
        sa.Column("cell_c_angstrom", sa.Float(), nullable=True),
        sa.Column("cell_alpha_deg", sa.Float(), nullable=True),
        sa.Column("cell_beta_deg", sa.Float(), nullable=True),
        sa.Column("cell_gamma_deg", sa.Float(), nullable=True),
        sa.Column("cell_volume_angstrom3", sa.Float(), nullable=True),
        sa.Column("has_coordinates", sa.Boolean(), nullable=False),
        sa.Column(
            "created_at", schema_types.UTCDateTime(timezone=True), nullable=False
        ),
        sa.Column(
            "updated_at", schema_types.UTCDateTime(timezone=True), nullable=False
        ),
        sa.CheckConstraint(
            "format IN ('cif', 'sdf', 'xyz')", name="ck_structures_format"
        ),
        sa.ForeignKeyConstraint(
            ["adsorbent_id"], ["adsorbents.id"], ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_structures_adsorbent", "structures", ["adsorbent_id"], unique=False
    )
    op.create_index("ix_structures_formula", "structures", ["formula"], unique=False)
    op.create_index("ix_structures_updated", "structures", ["updated_at"], unique=False)
    op.create_table(
        "training_snapshot_rows",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("snapshot_id", sa.String(length=36), nullable=False),
        sa.Column("row_index", sa.Integer(), nullable=False),
        sa.Column("payload", schema_types.JSONMapping(), nullable=False),
        sa.CheckConstraint("row_index >= 0", name="ck_training_snapshot_rows_index"),
        sa.ForeignKeyConstraint(
            ["snapshot_id"], ["training_snapshots.snapshot_id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "snapshot_id", "row_index", name="uq_training_snapshot_rows_index"
        ),
    )
    op.create_index(
        "ix_training_snapshot_rows_snapshot_index",
        "training_snapshot_rows",
        ["snapshot_id", "row_index"],
        unique=False,
    )
    op.create_table(
        "adsorbate_source_records",
        sa.Column("source_record_id", sa.Integer(), nullable=False),
        sa.Column("adsorbate_id", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["adsorbate_id"], ["adsorbates.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["source_record_id"], ["source_records.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("source_record_id"),
        sa.UniqueConstraint(
            "adsorbate_id", "source_record_id", name="uq_adsorbate_source_record"
        ),
    )
    op.create_index(
        "ix_adsorbate_source_records_adsorbate",
        "adsorbate_source_records",
        ["adsorbate_id"],
        unique=False,
    )
    op.create_table(
        "adsorbate_synonyms",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("adsorbate_id", sa.Integer(), nullable=False),
        sa.Column("source_record_id", sa.Integer(), nullable=True),
        sa.Column(
            "origin", sa.String(length=16), server_default="pubchem", nullable=False
        ),
        sa.Column("synonym", sa.String(length=1024), nullable=False),
        sa.Column("normalized_synonym", sa.String(length=1024), nullable=False),
        sa.ForeignKeyConstraint(
            ["adsorbate_id"], ["adsorbates.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["source_record_id"], ["source_records.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "adsorbate_id",
            "source_record_id",
            "normalized_synonym",
            name="uq_adsorbate_synonyms_identity",
        ),
    )
    op.create_index(
        "ix_adsorbate_synonyms_normalized",
        "adsorbate_synonyms",
        ["normalized_synonym"],
        unique=False,
    )
    op.create_index(
        "uq_adsorbate_synonyms_local",
        "adsorbate_synonyms",
        ["adsorbate_id", "origin", "normalized_synonym"],
        unique=True,
        sqlite_where=sa.text("source_record_id IS NULL"),
        postgresql_where=sa.text("source_record_id IS NULL"),
    )
    op.create_table(
        "adsorbent_source_records",
        sa.Column("source_record_id", sa.Integer(), nullable=False),
        sa.Column("adsorbent_id", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["adsorbent_id"], ["adsorbents.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["source_record_id"], ["source_records.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("source_record_id"),
        sa.UniqueConstraint(
            "adsorbent_id", "source_record_id", name="uq_adsorbent_source_record"
        ),
    )
    op.create_index(
        "ix_adsorbent_source_records_adsorbent",
        "adsorbent_source_records",
        ["adsorbent_id"],
        unique=False,
    )
    op.create_table(
        "adsorbent_synonyms",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("adsorbent_id", sa.Integer(), nullable=False),
        sa.Column("source_record_id", sa.Integer(), nullable=True),
        sa.Column("origin", sa.String(length=16), nullable=False),
        sa.Column("synonym", sa.String(length=1024), nullable=False),
        sa.Column("normalized_synonym", sa.String(length=1024), nullable=False),
        sa.ForeignKeyConstraint(
            ["adsorbent_id"], ["adsorbents.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["source_record_id"], ["source_records.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "adsorbent_id",
            "source_record_id",
            "normalized_synonym",
            name="uq_adsorbent_synonyms_identity",
        ),
    )
    op.create_index(
        "ix_adsorbent_synonyms_normalized",
        "adsorbent_synonyms",
        ["normalized_synonym"],
        unique=False,
    )
    op.create_index(
        "uq_adsorbent_synonyms_local",
        "adsorbent_synonyms",
        ["adsorbent_id", "origin", "normalized_synonym"],
        unique=True,
        sqlite_where=sa.text("source_record_id IS NULL"),
        postgresql_where=sa.text("source_record_id IS NULL"),
    )
    op.create_table(
        "chemical_properties",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("adsorbate_id", sa.Integer(), nullable=False),
        sa.Column("source_record_id", sa.Integer(), nullable=False),
        sa.Column("key", sa.String(length=128), nullable=False),
        sa.Column("value_number", sa.Float(), nullable=True),
        sa.Column("value_text", sa.String(length=2048), nullable=True),
        sa.Column("unit", sa.String(length=64), nullable=True),
        sa.CheckConstraint(
            "value_number IS NOT NULL OR value_text IS NOT NULL",
            name="ck_chemical_properties_value",
        ),
        sa.ForeignKeyConstraint(
            ["adsorbate_id"], ["adsorbates.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["source_record_id"], ["source_records.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "adsorbate_id", "source_record_id", "key", name="uq_chemical_properties_key"
        ),
    )
    op.create_index(
        "ix_chemical_properties_adsorbate_key",
        "chemical_properties",
        ["adsorbate_id", "key"],
        unique=False,
    )
    op.create_table(
        "fit_results",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("session_id", sa.Integer(), nullable=False),
        sa.Column("isotherm_id", sa.Integer(), nullable=True),
        sa.Column("isotherm_name", sa.String(length=255), nullable=False),
        sa.Column("model_name", sa.String(length=128), nullable=False),
        sa.Column("model_version", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("convergence_message", sa.String(), nullable=False),
        sa.Column("function_evaluations", sa.Integer(), nullable=True),
        sa.Column("jacobian_rank", sa.Integer(), nullable=True),
        sa.Column("condition_number", sa.Float(), nullable=True),
        sa.Column("observation_count", sa.Integer(), nullable=False),
        sa.Column("parameter_count", sa.Integer(), nullable=False),
        sa.Column("sse", sa.Float(), nullable=True),
        sa.Column("rmse", sa.Float(), nullable=True),
        sa.Column("mae", sa.Float(), nullable=True),
        sa.Column("r_squared", sa.Float(), nullable=True),
        sa.Column("adjusted_r_squared", sa.Float(), nullable=True),
        sa.Column("chi_square", sa.Float(), nullable=True),
        sa.Column("aic", sa.Float(), nullable=True),
        sa.Column("aicc", sa.Float(), nullable=True),
        sa.Column("bic", sa.Float(), nullable=True),
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
        sa.ForeignKeyConstraint(["isotherm_id"], ["isotherms.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(
            ["session_id"], ["fitting_sessions.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
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
        unique=False,
    )
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
        "ix_fitting_session_models_key",
        "fitting_session_models",
        ["model_key"],
        unique=False,
    )
    op.create_table(
        "isotherm_components",
        sa.Column("isotherm_id", sa.Integer(), nullable=False),
        sa.Column("adsorbate_id", sa.Integer(), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("source_name", sa.String(length=255), nullable=True),
        sa.CheckConstraint("position >= 1", name="ck_components_position"),
        sa.ForeignKeyConstraint(
            ["adsorbate_id"], ["adsorbates.id"], ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(["isotherm_id"], ["isotherms.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("isotherm_id", "adsorbate_id"),
        sa.UniqueConstraint("isotherm_id", "position", name="uq_components_position"),
    )
    op.create_index(
        "ix_components_adsorbate", "isotherm_components", ["adsorbate_id"], unique=False
    )
    op.create_table(
        "isotherm_source_records",
        sa.Column("source_record_id", sa.Integer(), nullable=False),
        sa.Column("isotherm_id", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(["isotherm_id"], ["isotherms.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["source_record_id"], ["source_records.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("source_record_id"),
        sa.UniqueConstraint(
            "isotherm_id", "source_record_id", name="uq_isotherm_source_record"
        ),
    )
    op.create_index(
        "ix_isotherm_source_records_isotherm",
        "isotherm_source_records",
        ["isotherm_id"],
        unique=False,
    )
    op.create_table(
        "public_data_sync_failures",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("sync_run_id", sa.String(length=36), nullable=False),
        sa.Column("source_key", sa.String(length=64), nullable=False),
        sa.Column("external_id", sa.String(length=512), nullable=True),
        sa.Column("reason", sa.String(length=2000), nullable=False),
        sa.Column(
            "retry_eligible_at", schema_types.UTCDateTime(timezone=True), nullable=True
        ),
        sa.Column(
            "created_at", schema_types.UTCDateTime(timezone=True), nullable=False
        ),
        sa.ForeignKeyConstraint(
            ["source_key"], ["data_sources.key"], ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(
            ["sync_run_id"], ["public_data_sync_runs.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_public_data_sync_failures_retry",
        "public_data_sync_failures",
        ["source_key", "retry_eligible_at"],
        unique=False,
    )
    op.create_index(
        "ix_public_data_sync_failures_run",
        "public_data_sync_failures",
        ["sync_run_id", "id"],
        unique=False,
    )
    op.create_table(
        "source_record_references",
        sa.Column("source_record_id", sa.Integer(), nullable=False),
        sa.Column("reference_id", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["reference_id"], ["references.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["source_record_id"], ["source_records.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("source_record_id", "reference_id"),
    )
    op.create_table(
        "structure_atoms",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("structure_id", sa.Integer(), nullable=False),
        sa.Column("sequence_index", sa.Integer(), nullable=False),
        sa.Column("label", sa.String(length=128), nullable=False),
        sa.Column("element", sa.String(length=8), nullable=False),
        sa.Column("fractional_x", sa.Float(), nullable=False),
        sa.Column("fractional_y", sa.Float(), nullable=False),
        sa.Column("fractional_z", sa.Float(), nullable=False),
        sa.Column("occupancy", sa.Float(), nullable=True),
        sa.CheckConstraint(
            "occupancy IS NULL OR (occupancy >= 0 AND occupancy <= 1)",
            name="ck_structure_atoms_occupancy",
        ),
        sa.CheckConstraint("sequence_index >= 0", name="ck_structure_atoms_sequence"),
        sa.ForeignKeyConstraint(
            ["structure_id"], ["structures.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "structure_id", "sequence_index", name="uq_structure_atoms_sequence"
        ),
    )
    op.create_index(
        "ix_structure_atoms_structure",
        "structure_atoms",
        ["structure_id", "sequence_index"],
        unique=False,
    )
    op.create_table(
        "structure_source_records",
        sa.Column("source_record_id", sa.Integer(), nullable=False),
        sa.Column("structure_id", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["source_record_id"], ["source_records.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["structure_id"], ["structures.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("source_record_id"),
        sa.UniqueConstraint(
            "structure_id", "source_record_id", name="uq_structure_source_record"
        ),
    )
    op.create_index(
        "ix_structure_source_records_structure",
        "structure_source_records",
        ["structure_id"],
        unique=False,
    )
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
    op.create_table(
        "observations",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("isotherm_id", sa.Integer(), nullable=False),
        sa.Column("adsorbate_id", sa.Integer(), nullable=False),
        sa.Column("source_row", sa.Integer(), nullable=True),
        sa.Column("pressure_original", sa.Float(), nullable=False),
        sa.Column("pressure_original_unit", sa.String(length=32), nullable=False),
        sa.Column("pressure_canonical", sa.Float(), nullable=False),
        sa.Column("pressure_canonical_unit", sa.String(length=8), nullable=False),
        sa.Column("pressure_key", sa.BigInteger(), nullable=False),
        sa.Column("uptake_original", sa.Float(), nullable=False),
        sa.Column("uptake_original_unit", sa.String(length=32), nullable=False),
        sa.Column("uptake_mol_kg", sa.Float(), nullable=False),
        sa.Column("uptake_stddev_mol_kg", sa.Float(), nullable=True),
        sa.Column("gas_mole_fraction", sa.Float(), nullable=True),
        sa.Column("conversion_metadata", schema_types.JSONMapping(), nullable=False),
        sa.Column("extra_metadata", schema_types.JSONMapping(), nullable=False),
        sa.CheckConstraint(
            "pressure_canonical_unit IN ('Pa', '1')",
            name="ck_observations_pressure_unit",
        ),
        sa.CheckConstraint(
            "gas_mole_fraction IS NULL OR (gas_mole_fraction >= 0 AND gas_mole_fraction <= 1)",
            name="ck_observations_gas_fraction",
        ),
        sa.CheckConstraint("pressure_canonical >= 0", name="ck_observations_pressure"),
        sa.CheckConstraint("uptake_mol_kg >= 0", name="ck_observations_uptake"),
        sa.CheckConstraint(
            "uptake_stddev_mol_kg IS NULL OR uptake_stddev_mol_kg > 0",
            name="ck_observations_uncertainty",
        ),
        sa.ForeignKeyConstraint(
            ["isotherm_id", "adsorbate_id"],
            ["isotherm_components.isotherm_id", "isotherm_components.adsorbate_id"],
            name="fk_observations_component",
            onupdate="CASCADE",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(["isotherm_id"], ["isotherms.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "isotherm_id",
            "adsorbate_id",
            "pressure_key",
            name="uq_observations_identity",
        ),
    )
    op.create_index(
        "ix_observations_species_pressure",
        "observations",
        ["isotherm_id", "adsorbate_id", "pressure_canonical"],
        unique=False,
    )


def downgrade() -> None:  # noqa: PLR0915
    op.drop_index("ix_observations_species_pressure", table_name="observations")
    op.drop_table("observations")
    op.drop_table("fit_parameters")
    op.drop_index(
        "ix_structure_source_records_structure", table_name="structure_source_records"
    )
    op.drop_table("structure_source_records")
    op.drop_index("ix_structure_atoms_structure", table_name="structure_atoms")
    op.drop_table("structure_atoms")
    op.drop_table("source_record_references")
    op.drop_index(
        "ix_public_data_sync_failures_run", table_name="public_data_sync_failures"
    )
    op.drop_index(
        "ix_public_data_sync_failures_retry", table_name="public_data_sync_failures"
    )
    op.drop_table("public_data_sync_failures")
    op.drop_index(
        "ix_isotherm_source_records_isotherm", table_name="isotherm_source_records"
    )
    op.drop_table("isotherm_source_records")
    op.drop_index("ix_components_adsorbate", table_name="isotherm_components")
    op.drop_table("isotherm_components")
    op.drop_index("ix_fitting_session_models_key", table_name="fitting_session_models")
    op.drop_table("fitting_session_models")
    op.drop_index("ix_fit_results_session_rank", table_name="fit_results")
    op.drop_table("fit_results")
    op.drop_index(
        "ix_chemical_properties_adsorbate_key", table_name="chemical_properties"
    )
    op.drop_table("chemical_properties")
    op.drop_index(
        "uq_adsorbent_synonyms_local",
        table_name="adsorbent_synonyms",
        sqlite_where=sa.text("source_record_id IS NULL"),
        postgresql_where=sa.text("source_record_id IS NULL"),
    )
    op.drop_index("ix_adsorbent_synonyms_normalized", table_name="adsorbent_synonyms")
    op.drop_table("adsorbent_synonyms")
    op.drop_index(
        "ix_adsorbent_source_records_adsorbent", table_name="adsorbent_source_records"
    )
    op.drop_table("adsorbent_source_records")
    op.drop_index(
        "uq_adsorbate_synonyms_local",
        table_name="adsorbate_synonyms",
        sqlite_where=sa.text("source_record_id IS NULL"),
        postgresql_where=sa.text("source_record_id IS NULL"),
    )
    op.drop_index("ix_adsorbate_synonyms_normalized", table_name="adsorbate_synonyms")
    op.drop_table("adsorbate_synonyms")
    op.drop_index(
        "ix_adsorbate_source_records_adsorbate", table_name="adsorbate_source_records"
    )
    op.drop_table("adsorbate_source_records")
    op.drop_index(
        "ix_training_snapshot_rows_snapshot_index", table_name="training_snapshot_rows"
    )
    op.drop_table("training_snapshot_rows")
    op.drop_index("ix_structures_updated", table_name="structures")
    op.drop_index("ix_structures_formula", table_name="structures")
    op.drop_index("ix_structures_adsorbent", table_name="structures")
    op.drop_table("structures")
    op.drop_index("ix_source_records_source_retrieved", table_name="source_records")
    op.drop_table("source_records")
    op.drop_index(
        "ix_public_data_sync_runs_source_created", table_name="public_data_sync_runs"
    )
    op.drop_index("ix_public_data_sync_runs_queue", table_name="public_data_sync_runs")
    op.drop_table("public_data_sync_runs")
    op.drop_index("ix_isotherms_temperature", table_name="isotherms")
    op.drop_index("ix_isotherms_dataset_name", table_name="isotherms")
    op.drop_index("ix_isotherms_dataset_edited", table_name="isotherms")
    op.drop_index("ix_isotherms_adsorbent", table_name="isotherms")
    op.drop_table("isotherms")
    op.drop_index("ix_fitting_sessions_dataset", table_name="fitting_sessions")
    op.drop_index("ix_fitting_sessions_created", table_name="fitting_sessions")
    op.drop_table("fitting_sessions")
    op.drop_table("dataset_imports")
    op.drop_table("training_snapshots")
    op.drop_index("ix_references_year", table_name="references")
    op.drop_table("references")
    op.drop_index("ix_identity_reviews_status", table_name="identity_reviews")
    op.drop_table("identity_reviews")
    op.drop_index("ix_datasets_source_created_at", table_name="datasets")
    op.drop_table("datasets")
    op.drop_table("data_sources")
    op.drop_table("custom_units")
    op.drop_table("application_settings")
    op.drop_table("adsorption_model_settings")
    op.drop_index("ix_adsorbents_normalized_name", table_name="adsorbents")
    op.drop_index("ix_adsorbents_formula", table_name="adsorbents")
    op.drop_table("adsorbents")
    op.drop_index("ix_adsorbates_normalized_name", table_name="adsorbates")
    op.drop_index("ix_adsorbates_molar_mass", table_name="adsorbates")
    op.drop_index("ix_adsorbates_formula", table_name="adsorbates")
    op.drop_table("adsorbates")
