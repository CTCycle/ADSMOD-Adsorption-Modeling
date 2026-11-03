# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

"""Add persisted user-defined units.

Revision ID: 20261103_custom_units
Revises: 20261021_current_schema
"""

from alembic import op
import sqlalchemy as sa

from server.repositories.schemas import types as schema_types


revision = "20261103_custom_units"
down_revision = "20261021_current_schema"
branch_labels = None
depends_on = None


def upgrade() -> None:
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
            "factor_to_canonical IS NULL OR factor_to_canonical > 0",
            name="ck_custom_units_factor",
        ),
        sa.CheckConstraint(
            "length(normalized_symbol) > 0", name="ck_custom_units_symbol"
        ),
        sa.CheckConstraint(
            "quantity IN ('pressure', 'uptake', 'temperature')",
            name="ck_custom_units_quantity",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "quantity", "normalized_symbol", name="uq_custom_units_quantity_symbol"
        ),
    )


def downgrade() -> None:
    op.drop_table("custom_units")
