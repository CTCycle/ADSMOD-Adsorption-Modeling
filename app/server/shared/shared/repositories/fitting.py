from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from sqlalchemy import select

from shared.repositories.database.bulk import upsert_records
from shared.repositories.database.manager import DatabaseManager
from shared.repositories.schemas.models import Fit, FitParameter, ProcessedIsotherm


###############################################################################
class FittingRepository:

    # -------------------------------------------------------------------------
    def __init__(self, database: DatabaseManager) -> None:
        self.database = database

    # -------------------------------------------------------------------------
    def upsert_processed(self, records: Iterable[dict[str, Any]]) -> int:
        with self.database.transaction() as session:
            return upsert_records(session, ProcessedIsotherm.__table__, records, ["isotherm_id", "processing_version"])

    # -------------------------------------------------------------------------
    def upsert_fits(self, records: Iterable[dict[str, Any]]) -> int:
        with self.database.transaction() as session:
            return upsert_records(session, Fit.__table__, records, ["processed_isotherm_id", "model_name", "model_version", "optimization_method"])

    # -------------------------------------------------------------------------
    def upsert_parameters(self, records: Iterable[dict[str, Any]]) -> int:
        with self.database.transaction() as session:
            return upsert_records(session, FitParameter.__table__, records, ["fit_id", "parameter_name"])

    # -------------------------------------------------------------------------
    def ranked_fits(self, processed_id: int, metric: str = "aicc") -> list[Fit]:
        if metric not in {"aicc", "objective_score"}:
            raise ValueError("Unsupported fit ranking metric.")
        with self.database.session_factory() as session:
            column = getattr(Fit, metric)
            return list(session.scalars(select(Fit).where(Fit.processed_isotherm_id == processed_id).order_by(column.is_(None), column.asc(), Fit.id)))
