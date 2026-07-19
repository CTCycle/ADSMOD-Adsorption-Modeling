from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from sqlalchemy import select

from shared.repositories.database.bulk import upsert_records
from shared.repositories.database.manager import DatabaseManager
from shared.repositories.schemas.models import Isotherm, IsothermComponent, IsothermMeasurement

###############################################################################
class IsothermRepository:

    # -------------------------------------------------------------------------
    def __init__(self, database: DatabaseManager) -> None:
        self.database = database

    # -------------------------------------------------------------------------
    def upsert_isotherms(self, records: Iterable[dict[str, Any]]) -> int:
        with self.database.transaction() as session:
            return upsert_records(session, Isotherm.__table__, records, ["isotherm_key"])

    # -------------------------------------------------------------------------
    def upsert_components(self, records: Iterable[dict[str, Any]]) -> int:
        with self.database.transaction() as session:
            return upsert_records(session, IsothermComponent.__table__, records, ["isotherm_id", "position"])

    # -------------------------------------------------------------------------
    def upsert_measurements(self, records: Iterable[dict[str, Any]]) -> int:
        with self.database.transaction() as session:
            return upsert_records(session, IsothermMeasurement.__table__, records, ["isotherm_id", "point_index", "component_id"])

    # -------------------------------------------------------------------------
    def measurement_page(self, isotherm_id: int, offset: int = 0, limit: int = 100) -> list[IsothermMeasurement]:
        with self.database.session_factory() as session:
            return list(session.scalars(select(IsothermMeasurement).where(IsothermMeasurement.isotherm_id == isotherm_id).order_by(IsothermMeasurement.point_index, IsothermMeasurement.id).offset(offset).limit(limit)))
