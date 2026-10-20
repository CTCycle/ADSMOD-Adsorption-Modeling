# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

"""Normalize public-data identities, provenance, and navigation indexes.

Revision ID: 20261020_public_data_nav
Revises: 20261019_public_data_sync
"""

from __future__ import annotations

from typing import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "20261020_public_data_nav"
down_revision: str | None = "20261019_public_data_sync"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

NIST_ISOTHERM_URL = "https://adsorption.nist.gov/isodb/api/isotherm/{}.json"

###############################################################################
def _referencing_columns(bind: sa.Connection, table: str) -> list[tuple[str, str]]:
    inspector = sa.inspect(bind)
    columns: list[tuple[str, str]] = []
    for name in inspector.get_table_names():
        for foreign_key in inspector.get_foreign_keys(name):
            if foreign_key["referred_table"] == table:
                columns.extend(
                    (name, column) for column in foreign_key["constrained_columns"]
                )
    return columns

###############################################################################
def _merge_rows(
    bind: sa.Connection, table: str, duplicate_id: int, canonical_id: int
) -> None:
    for child_table, column in _referencing_columns(bind, table):
        bind.execute(
            sa.text(
                f'UPDATE "{child_table}" SET "{column}" = :canonical '
                f'WHERE "{column}" = :duplicate'
            ),
            {"canonical": canonical_id, "duplicate": duplicate_id},
        )
    bind.execute(
        sa.text(f'DELETE FROM "{table}" WHERE id = :duplicate'),
        {"duplicate": duplicate_id},
    )

###############################################################################
def _remove_placeholder_identifiers(bind: sa.Connection) -> None:
    # The 20260902 backfill used internal keys where NIST had no identifier.
    # Those rows are not provider identities and are removed with their links.
    for link_table, entity_table, record_type in (
        ("adsorbate_source_records", "adsorbates", "chemical"),
        ("adsorbent_source_records", "adsorbents", "material"),
    ):
        placeholder_ids = [
            row[0]
            for row in bind.execute(
                sa.text(
                    f"""
                    SELECT sr.id
                    FROM source_records AS sr
                    JOIN {link_table} AS link ON link.source_record_id = sr.id
                    JOIN {entity_table} AS entity
                        ON entity.id = link.{entity_table[:-1]}_id
                    WHERE sr.record_type = :record_type
                      AND sr.external_id = entity.key
                    """
                ),
                {"record_type": record_type},
            )
        ]
        for record_id in placeholder_ids:
            for table in (link_table, "source_record_references"):
                bind.execute(
                    sa.text(f"DELETE FROM {table} WHERE source_record_id = :id"),
                    {"id": record_id},
                )
            bind.execute(
                sa.text("DELETE FROM source_records WHERE id = :id"), {"id": record_id}
            )

###############################################################################
def _merge_nist_duplicates(
    bind: sa.Connection,
    *,
    entity_table: str,
    link_table: str,
    identifier_column: str,
    usage_sql: str,
) -> None:
    """Merge NIST isotherm rows into the NIST catalogue row of the same entity.

    NIST experiments used to persist species and materials by display name while
    the NIST catalogue keys them by InChIKey or hashkey. A row is merged only when
    it has no identifier or provenance of its own, is used solely by NIST
    isotherms, and its normalized name matches exactly one NIST catalogue row.
    """

    entity_column = f"{entity_table[:-1]}_id"
    candidates = bind.execute(
        sa.text(
            f"""
            SELECT e.id, e.normalized_name
            FROM {entity_table} AS e
            WHERE e.{identifier_column} IS NULL
              AND NOT EXISTS (
                  SELECT 1 FROM {link_table} AS link WHERE link.{entity_column} = e.id
              )
              AND EXISTS ({usage_sql} AND d.source = 'nist')
              AND NOT EXISTS ({usage_sql} AND d.source <> 'nist')
            """
        )
    ).all()
    for duplicate_id, normalized_name in candidates:
        matches = bind.execute(
            sa.text(
                f"""
                SELECT DISTINCT e.id
                FROM {entity_table} AS e
                JOIN {link_table} AS link ON link.{entity_column} = e.id
                JOIN source_records AS sr ON sr.id = link.source_record_id
                JOIN data_sources AS ds ON ds.id = sr.source_id
                WHERE ds.key = 'nist'
                  AND e.{identifier_column} IS NOT NULL
                  AND e.normalized_name = :name
                """
            ),
            {"name": normalized_name},
        ).all()
        if len(matches) == 1:
            _merge_rows(bind, entity_table, int(duplicate_id), int(matches[0][0]))

###############################################################################
def upgrade() -> None:
    bind = op.get_bind()

    # NIST placeholder identifiers ("NIST-ISODB-...") are not InChIKeys; they
    # remain on the NIST source records and stop blocking PubChem resolution.
    bind.execute(
        sa.text(
            "UPDATE adsorbates SET inchi_key = NULL "
            "WHERE inchi_key IS NOT NULL AND length(inchi_key) <> 27"
        )
    )
    _remove_placeholder_identifiers(bind)
    _merge_nist_duplicates(
        bind,
        entity_table="adsorbates",
        link_table="adsorbate_source_records",
        identifier_column="inchi_key",
        usage_sql=(
            "SELECT 1 FROM isotherm_components AS c "
            "JOIN isotherms AS i ON i.id = c.isotherm_id "
            "JOIN datasets AS d ON d.id = i.dataset_id "
            "WHERE c.adsorbate_id = e.id"
        ),
    )
    _merge_nist_duplicates(
        bind,
        entity_table="adsorbents",
        link_table="adsorbent_source_records",
        identifier_column="external_identifier",
        usage_sql=(
            "SELECT 1 FROM isotherms AS i "
            "JOIN datasets AS d ON d.id = i.dataset_id "
            "WHERE i.adsorbent_id = e.id"
        ),
    )
    for record_id, external_id in bind.execute(
        sa.text(
            """
            SELECT sr.id, sr.external_id
            FROM source_records AS sr
            JOIN data_sources AS ds ON ds.id = sr.source_id
            WHERE ds.key = 'nist' AND sr.record_type = 'adsorption'
            """
        )
    ).all():
        bind.execute(
            sa.text("UPDATE source_records SET source_url = :url WHERE id = :id"),
            {"url": NIST_ISOTHERM_URL.format(external_id), "id": record_id},
        )

    op.drop_table("material_properties")
    with op.batch_alter_table("data_sources") as batch_op:
        batch_op.drop_column("enabled")
    with op.batch_alter_table("source_records") as batch_op:
        batch_op.drop_column("transform_version")
        batch_op.drop_constraint("ck_source_records_type", type_="check")
        batch_op.create_check_constraint(
            "ck_source_records_type",
            "record_type IN ('adsorption', 'chemical', 'material', 'structure')",
        )
    with op.batch_alter_table("public_data_sync_runs") as batch_op:
        batch_op.drop_constraint("ck_public_data_sync_runs_operation", type_="check")
        batch_op.create_check_constraint(
            "ck_public_data_sync_runs_operation",
            "operation IN ('health_check', 'nist_index', 'nist_category', "
            "'pubchem_resolve', 'pubchem_enrich', 'cod_import')",
        )

    op.create_index("ix_adsorbates_formula", "adsorbates", ["formula"])
    op.create_index("ix_adsorbates_molar_mass", "adsorbates", ["molar_mass_g_mol"])
    op.create_index("ix_adsorbents_formula", "adsorbents", ["formula"])
    op.create_index("ix_isotherms_adsorbent", "isotherms", ["adsorbent_id"])
    op.create_index("ix_isotherms_temperature", "isotherms", ["temperature_k"])
    op.create_index("ix_components_adsorbate", "isotherm_components", ["adsorbate_id"])
    op.create_index("ix_structures_updated", "structures", ["updated_at"])

###############################################################################
def downgrade() -> None:
    raise NotImplementedError(
        "Public-data identity merges cannot be reversed; restore a backup instead."
    )
