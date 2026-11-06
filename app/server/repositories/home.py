# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

from typing import Any

from sqlalchemy import func, select

from server.repositories.database.manager import DatabaseManager
from server.repositories.schemas.models import (
    Adsorbate,
    Adsorbent,
    Dataset,
    FittingSession,
    Isotherm,
)
from server.repositories.schemas.public_data import DataSource, SourceRecord


###############################################################################
class HomeRepository:
    """Read-only totals for the Home overview."""

    # -------------------------------------------------------------------------
    def __init__(self, database: DatabaseManager) -> None:
        self.database = database

    # -------------------------------------------------------------------------
    def summary(self) -> dict[str, Any]:
        with self.database.session_factory() as session:
            totals = session.execute(
                select(
                    select(func.count(Dataset.id)).scalar_subquery().label("datasets"),
                    select(func.count(Isotherm.id)).scalar_subquery().label("experiments"),
                    select(func.count(Adsorbate.id)).scalar_subquery().label("species"),
                    select(func.count(Adsorbent.id)).scalar_subquery().label("materials"),
                )
            ).one()
            sources = session.execute(
                select(DataSource.key, DataSource.name, func.count(SourceRecord.id))
                .join(SourceRecord, SourceRecord.source_id == DataSource.id)
                .group_by(DataSource.id, DataSource.key, DataSource.name)
                .order_by(DataSource.name)
            ).all()
            by_status = dict(
                session.execute(
                    select(FittingSession.status, func.count(FittingSession.id)).group_by(
                        FittingSession.status
                    )
                ).all()
            )
            latest = session.scalar(select(func.max(FittingSession.created_at)))
        return {
            "datasets": int(totals.datasets),
            "experiments": int(totals.experiments),
            "species": int(totals.species),
            "materials": int(totals.materials),
            "sources": [
                {"key": key, "name": name, "record_count": int(count)}
                for key, name, count in sources
            ],
            "fitting_sessions": int(sum(by_status.values())),
            "fitting_sessions_by_status": {k: int(v) for k, v in by_status.items()},
            "latest_fitting_at": latest.isoformat() if latest else None,
        }


__all__ = ["HomeRepository"]
