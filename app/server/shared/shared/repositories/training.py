from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from sqlalchemy import select

from shared.repositories.database.bulk import upsert_records
from shared.repositories.database.manager import DatabaseManager
from shared.repositories.schemas.models import TrainingDataset, TrainingSample

###############################################################################
class TrainingRepository:

    # -------------------------------------------------------------------------
    def __init__(self, database: DatabaseManager) -> None:
        self.database = database

    # -------------------------------------------------------------------------
    def create_dataset(self, record: dict[str, Any]) -> int:
        with self.database.transaction() as session:
            dataset = TrainingDataset(**record)
            session.add(dataset)
            session.flush()
            return dataset.id

    # -------------------------------------------------------------------------
    def upsert_samples(self, records: Iterable[dict[str, Any]]) -> int:
        with self.database.transaction() as session:
            return upsert_records(session, TrainingSample.__table__, records, ["training_dataset_id", "sample_key"])

    # -------------------------------------------------------------------------
    def samples(self, dataset_id: int, split: str | None = None, offset: int = 0, limit: int = 1000) -> list[TrainingSample]:
        statement = select(TrainingSample).where(TrainingSample.training_dataset_id == dataset_id).order_by(TrainingSample.id).offset(offset).limit(limit)
        if split:
            statement = statement.where(TrainingSample.split == split)
        with self.database.session_factory() as session:
            return list(session.scalars(statement))
