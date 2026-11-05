# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

"""Track local experiment edits.

Revision ID: 20261105_dataset_editing
Revises: 20261103_custom_units
"""

from alembic import op
import sqlalchemy as sa

from server.repositories.schemas import types as schema_types


revision = "20261105_dataset_editing"
down_revision = "20261103_custom_units"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("isotherms") as batch:
        batch.add_column(
            sa.Column(
                "edited_at", schema_types.UTCDateTime(timezone=True), nullable=True
            )
        )
    op.create_index(
        "ix_isotherms_dataset_edited", "isotherms", ["dataset_id", "edited_at"]
    )


def downgrade() -> None:
    op.drop_index("ix_isotherms_dataset_edited", table_name="isotherms")
    with op.batch_alter_table("isotherms") as batch:
        batch.drop_column("edited_at")
