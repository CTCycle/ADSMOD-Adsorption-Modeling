from __future__ import annotations

from sqlalchemy import select

from shared.repositories.database.manager import DatabaseManager
from shared.repositories.schemas.models import Observation

###############################################################################
class IsothermRepository:
    """Read-only atomic-observation access outside the dataset aggregate."""

    # -------------------------------------------------------------------------
    def __init__(self, database: DatabaseManager) -> None:
        self.database = database

    # -------------------------------------------------------------------------
    def observation_page(
        self, isotherm_id: int, offset: int = 0, limit: int = 100
    ) -> list[Observation]:
        with self.database.session_factory() as session:
            return list(
                session.scalars(
                    select(Observation)
                    .where(Observation.isotherm_id == isotherm_id)
                    .order_by(Observation.sequence_index, Observation.id)
                    .offset(offset)
                    .limit(limit)
                )
            )
