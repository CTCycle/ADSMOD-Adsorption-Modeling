# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

"""Add species identity status, aliases, source names, and the review queue.

Additive only: no canonical row is merged, split, or renamed here. Merges happen
through approved identity reviews.

Revision ID: 20261110_species_identity
Revises: 20261106_modeling_sessions
"""

from alembic import op
import sqlalchemy as sa

from server.common.identity import is_generic_adsorbent
from server.repositories.schemas import types as schema_types


revision = "20261110_species_identity"
down_revision = "20261106_modeling_sessions"
branch_labels = None
depends_on = None

LOCAL_ALIAS = sa.text("source_record_id IS NULL")


def upgrade() -> None:
    with op.batch_alter_table("adsorbates") as batch:
        batch.add_column(
            sa.Column(
                "identity_status",
                sa.String(16),
                nullable=False,
                server_default="provisional",
            )
        )
    with op.batch_alter_table("adsorbents") as batch:
        batch.add_column(
            sa.Column(
                "identity_status",
                sa.String(16),
                nullable=False,
                server_default="provisional",
            )
        )
        # A plain integer: a foreign key here would force SQLite to rebuild
        # adsorbents, and dropping that table fires its dependants' cascades.
        batch.add_column(sa.Column("scope_dataset_id", sa.Integer(), nullable=True))
    with op.batch_alter_table("isotherms") as batch:
        batch.add_column(sa.Column("adsorbent_source_name", sa.String(255)))
    with op.batch_alter_table("isotherm_components") as batch:
        batch.add_column(sa.Column("source_name", sa.String(255)))
    with op.batch_alter_table("application_settings") as batch:
        batch.add_column(
            sa.Column(
                "public_data_auto_resolve_uploaded_species",
                sa.Boolean(),
                nullable=False,
                server_default=sa.true(),
            )
        )
    with op.batch_alter_table("adsorbate_synonyms") as batch:
        batch.alter_column(
            "source_record_id", existing_type=sa.Integer(), nullable=True
        )
        batch.add_column(
            sa.Column(
                "origin", sa.String(16), nullable=False, server_default="pubchem"
            )
        )
    op.create_index(
        "uq_adsorbate_synonyms_local",
        "adsorbate_synonyms",
        ["adsorbate_id", "origin", "normalized_synonym"],
        unique=True,
        sqlite_where=LOCAL_ALIAS,
        postgresql_where=LOCAL_ALIAS,
    )

    op.create_table(
        "adsorbent_synonyms",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "adsorbent_id",
            sa.Integer(),
            sa.ForeignKey("adsorbents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "source_record_id",
            sa.Integer(),
            sa.ForeignKey("source_records.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column("origin", sa.String(16), nullable=False),
        sa.Column("synonym", sa.String(1024), nullable=False),
        sa.Column("normalized_synonym", sa.String(1024), nullable=False),
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
    )
    op.create_index(
        "uq_adsorbent_synonyms_local",
        "adsorbent_synonyms",
        ["adsorbent_id", "origin", "normalized_synonym"],
        unique=True,
        sqlite_where=LOCAL_ALIAS,
        postgresql_where=LOCAL_ALIAS,
    )

    op.create_table(
        "identity_reviews",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("entity_kind", sa.String(16), nullable=False),
        sa.Column("entity_id", sa.Integer(), nullable=True),
        sa.Column("candidate_id", sa.Integer(), nullable=True),
        sa.Column("raw_name", sa.String(1024), nullable=False),
        sa.Column("reason", sa.String(32), nullable=False),
        sa.Column("score", sa.Float(), nullable=True),
        sa.Column("evidence", schema_types.JSONMapping(), nullable=False),
        sa.Column("fingerprint", sa.String(128), nullable=False, unique=True),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("merge_snapshot", schema_types.JSONMapping(), nullable=True),
        sa.Column("created_at", schema_types.UTCDateTime(), nullable=False),
        sa.Column("decided_at", schema_types.UTCDateTime(), nullable=True),
        sa.CheckConstraint(
            "entity_kind IN ('adsorbate', 'adsorbent')",
            name="ck_identity_reviews_kind",
        ),
        sa.CheckConstraint(
            "status IN ('open', 'merged', 'rejected')",
            name="ck_identity_reviews_status",
        ),
    )
    op.create_index(
        "ix_identity_reviews_status", "identity_reviews", ["status", "entity_kind"]
    )
    _backfill()


def _backfill() -> None:
    bind = op.get_bind()
    bind.execute(
        sa.text(
            "UPDATE adsorbates SET identity_status = 'resolved' "
            "WHERE inchi_key IS NOT NULL OR id IN "
            "(SELECT adsorbate_id FROM adsorbate_source_records)"
        )
    )
    bind.execute(
        sa.text(
            "UPDATE adsorbents SET identity_status = 'resolved' "
            "WHERE external_identifier IS NOT NULL OR id IN "
            "(SELECT adsorbent_id FROM adsorbent_source_records)"
        )
    )
    bind.execute(
        sa.text(
            "UPDATE isotherms SET adsorbent_source_name = "
            "(SELECT name FROM adsorbents WHERE adsorbents.id = isotherms.adsorbent_id)"
        )
    )
    bind.execute(
        sa.text(
            "UPDATE isotherm_components SET source_name = (SELECT name FROM adsorbates "
            "WHERE adsorbates.id = isotherm_components.adsorbate_id)"
        )
    )
    provisional = bind.execute(
        sa.text("SELECT id, name FROM adsorbents WHERE identity_status = 'provisional'")
    ).all()
    generic = [{"id": row.id} for row in provisional if is_generic_adsorbent(row.name)]
    if generic:
        bind.execute(
            sa.text("UPDATE adsorbents SET identity_status = 'generic' WHERE id = :id"),
            generic,
        )


def downgrade() -> None:
    op.drop_table("identity_reviews")
    op.drop_table("adsorbent_synonyms")
    op.drop_index("uq_adsorbate_synonyms_local", table_name="adsorbate_synonyms")
    op.execute("DELETE FROM adsorbate_synonyms WHERE source_record_id IS NULL")
    with op.batch_alter_table("adsorbate_synonyms") as batch:
        batch.drop_column("origin")
        batch.alter_column(
            "source_record_id", existing_type=sa.Integer(), nullable=False
        )
    with op.batch_alter_table("application_settings") as batch:
        batch.drop_column("public_data_auto_resolve_uploaded_species")
    with op.batch_alter_table("isotherm_components") as batch:
        batch.drop_column("source_name")
    with op.batch_alter_table("isotherms") as batch:
        batch.drop_column("adsorbent_source_name")
    with op.batch_alter_table("adsorbents") as batch:
        batch.drop_column("scope_dataset_id")
        batch.drop_column("identity_status")
    with op.batch_alter_table("adsorbates") as batch:
        batch.drop_column("identity_status")
