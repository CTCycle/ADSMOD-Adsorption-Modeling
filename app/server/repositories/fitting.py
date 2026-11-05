# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal

from sqlalchemy import delete, func, select
from sqlalchemy.orm import selectinload

from server.repositories.database.manager import DatabaseManager
from server.repositories.schemas.models import (
    FitParameter,
    FitResult,
    FittingSession,
    FittingSessionModel,
)

SessionSort = Literal["created_at", "name", "status"]

###############################################################################
@dataclass(frozen=True)
class SessionQuery:
    query: str = ""
    statuses: tuple[str, ...] = ()
    dataset_id: int | None = None
    model: str | None = None
    created_from: datetime | None = None
    created_to: datetime | None = None
    sort: SessionSort = "created_at"
    descending: bool = True
    page: int = 1
    page_size: int = 20

###############################################################################
class FittingRepository:

    # -------------------------------------------------------------------------
    def __init__(self, database: DatabaseManager) -> None:
        self.database = database

    # -------------------------------------------------------------------------
    def create_session(
        self,
        *,
        name: str,
        dataset_id: int,
        dataset_name: str,
        isotherm_ids: list[int],
        models: list[str],
        input_sha256: str,
        optimizer: str,
        weighting: str,
        max_evaluations: int,
        pressure_display_unit: str,
        uptake_display_unit: str,
        configuration: dict[str, Any],
    ) -> int:
        with self.database.transaction() as session:
            record = FittingSession(
                name=name,
                dataset_id=dataset_id,
                dataset_name=dataset_name,
                isotherm_ids=isotherm_ids,
                input_sha256=input_sha256,
                optimizer=optimizer,
                weighting=weighting,
                max_evaluations=max_evaluations,
                pressure_display_unit=pressure_display_unit,
                uptake_display_unit=uptake_display_unit,
                configuration=configuration,
                status="running",
                log=[],
                models=[
                    FittingSessionModel(model_key=key, position=index)
                    for index, key in enumerate(models)
                ],
            )
            session.add(record)
            session.flush()
            return record.id

    # -------------------------------------------------------------------------
    def complete_session(
        self,
        session_id: int,
        *,
        status: str,
        message: str,
        results: list[dict[str, Any]],
        log: list[str],
    ) -> None:
        with self.database.transaction() as session:
            record = session.get(FittingSession, session_id)
            if record is None:
                raise LookupError(f"Fitting session {session_id} does not exist.")
            for result_record in results:
                values = dict(result_record)
                parameters = values.pop("parameters")
                result = FitResult(session_id=session_id, **values)
                session.add(result)
                session.flush()
                for parameter in parameters:
                    session.add(FitParameter(result_id=result.id, **parameter))
            record.status = status
            record.message = message
            record.log = log
            record.completed_at = datetime.now(timezone.utc)

    # -------------------------------------------------------------------------
    def _finish(
        self, session_id: int, status: str, message: str, log: list[str] | None
    ) -> bool:
        with self.database.transaction() as session:
            record = session.get(FittingSession, session_id)
            if record is None:
                return False
            if record.status != "running":
                if log is not None and record.status == status:
                    record.log = log
                return False
            record.status = status
            record.message = message
            if log is not None:
                record.log = log
            record.completed_at = datetime.now(timezone.utc)
            return True

    # -------------------------------------------------------------------------
    def fail_session(
        self, session_id: int, message: str, log: list[str] | None = None
    ) -> None:
        self._finish(session_id, "failed", message, log)

    # -------------------------------------------------------------------------
    def cancel_session(
        self,
        session_id: int,
        message: str = "Fitting cancelled.",
        log: list[str] | None = None,
    ) -> bool:
        return self._finish(session_id, "cancelled", message, log)

    # -------------------------------------------------------------------------
    def is_session_running(self, session_id: int) -> bool:
        with self.database.session_factory() as session:
            record = session.get(FittingSession, session_id)
            return record is not None and record.status == "running"

    # -------------------------------------------------------------------------
    def fail_interrupted_sessions(self, message: str) -> int:
        """Mark sessions left running by a previous process as failed."""
        with self.database.transaction() as session:
            records = session.scalars(
                select(FittingSession).where(FittingSession.status == "running")
            ).all()
            now = datetime.now(timezone.utc)
            for record in records:
                record.status = "failed"
                record.message = message
                record.completed_at = now
            return len(records)

    # -------------------------------------------------------------------------
    def get_session(self, session_id: int) -> FittingSession:
        with self.database.session_factory() as session:
            record = session.scalar(
                select(FittingSession)
                .where(FittingSession.id == session_id)
                .options(
                    selectinload(FittingSession.models),
                    selectinload(FittingSession.results).selectinload(
                        FitResult.parameters
                    ),
                )
            )
            if record is None:
                raise LookupError(f"Fitting session {session_id} does not exist.")
            return record

    # -------------------------------------------------------------------------
    def list_sessions(
        self, query: SessionQuery
    ) -> tuple[list[FittingSession], int]:
        conditions = []
        if query.query.strip():
            pattern = f"%{query.query.strip().casefold()}%"
            conditions.append(func.lower(FittingSession.name).like(pattern))
        if query.statuses:
            conditions.append(FittingSession.status.in_(query.statuses))
        if query.dataset_id is not None:
            conditions.append(FittingSession.dataset_id == query.dataset_id)
        if query.model:
            conditions.append(
                FittingSession.id.in_(
                    select(FittingSessionModel.session_id).where(
                        FittingSessionModel.model_key == query.model
                    )
                )
            )
        if query.created_from is not None:
            conditions.append(FittingSession.created_at >= query.created_from)
        if query.created_to is not None:
            conditions.append(FittingSession.created_at < query.created_to)
        column = {
            "created_at": FittingSession.created_at,
            "name": func.lower(FittingSession.name),
            "status": FittingSession.status,
        }[query.sort]
        order = column.desc() if query.descending else column.asc()
        tiebreak = FittingSession.id.desc() if query.descending else FittingSession.id.asc()
        with self.database.session_factory() as session:
            total = session.scalar(
                select(func.count(FittingSession.id)).where(*conditions)
            )
            records = session.scalars(
                select(FittingSession)
                .where(*conditions)
                .options(
                    selectinload(FittingSession.models),
                    selectinload(FittingSession.results),
                )
                .order_by(order, tiebreak)
                .offset((query.page - 1) * query.page_size)
                .limit(query.page_size)
            ).all()
            return list(records), int(total or 0)

    # -------------------------------------------------------------------------
    def delete_session(self, session_id: int) -> None:
        with self.database.transaction() as session:
            record = session.get(FittingSession, session_id)
            if record is None:
                raise LookupError(f"Fitting session {session_id} does not exist.")
            if record.status == "running":
                raise ValueError("A running fitting session cannot be deleted.")
            session.execute(
                delete(FittingSession).where(FittingSession.id == session_id)
            )
