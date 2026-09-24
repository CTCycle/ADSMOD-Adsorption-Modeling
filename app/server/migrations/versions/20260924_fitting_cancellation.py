"""Allow cancelled fitting runs to persist their terminal status.

Revision ID: 20260924_fitting_cancel
Revises: 20260902_public_data
"""

from __future__ import annotations

from typing import Sequence

from alembic import op


revision: str = "20260924_fitting_cancel"
down_revision: str | None = "20260902_public_data"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


###############################################################################
def upgrade() -> None:
    with op.batch_alter_table("fitting_runs") as batch_op:
        batch_op.drop_constraint("ck_fitting_runs_status", type_="check")
        batch_op.create_check_constraint(
            "ck_fitting_runs_status",
            "status IN ('running', 'completed', 'warning', 'failed', 'cancelled')",
        )


###############################################################################
def downgrade() -> None:
    with op.batch_alter_table("fitting_runs") as batch_op:
        batch_op.drop_constraint("ck_fitting_runs_status", type_="check")
        batch_op.create_check_constraint(
            "ck_fitting_runs_status",
            "status IN ('running', 'completed', 'warning', 'failed')",
        )
