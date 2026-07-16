from __future__ import annotations

from sqlalchemy import delete, func, select

from shared.repositories.database.manager import DatabaseManager
from shared.repositories.schemas.models import Dataset, Isotherm
from shared.repositories.schemas.types import normalize_identity


###############################################################################
class DatasetRepository:

    # -------------------------------------------------------------------------
    def __init__(self, database: DatabaseManager) -> None:
        self.database = database

    # -------------------------------------------------------------------------
    def create(self, name: str, source: str, description: str = "", tags: list[object] | None = None) -> Dataset:
        with self.database.transaction() as session:
            dataset = Dataset(name=name, source=source, description=description, tags=tags or [])
            session.add(dataset)
            session.flush()
            return dataset

    # -------------------------------------------------------------------------
    def list(self, *, source: str | None = None, offset: int = 0, limit: int = 100) -> list[tuple[int, str, str, int]]:
        statement = select(Dataset.id, Dataset.name, Dataset.source, func.count(Isotherm.id)).outerjoin(Isotherm, Isotherm.dataset_id == Dataset.id).where(Dataset.source == source if source else True).group_by(Dataset.id).order_by(Dataset.id).offset(offset).limit(limit)
        with self.database.session_factory() as session:
            return list(session.execute(statement).all())

    # -------------------------------------------------------------------------
    def count(self, *, source: str | None = None) -> int:
        with self.database.session_factory() as session:
            return int(session.scalar(select(func.count()).select_from(Dataset).where(Dataset.source == source if source else True)) or 0)

    # -------------------------------------------------------------------------
    def rename(self, dataset_id: int, name: str) -> None:
        with self.database.transaction() as session:
            dataset = session.get(Dataset, dataset_id)
            if dataset is None:
                raise LookupError(f"Dataset {dataset_id} does not exist.")
            dataset.name = name
            dataset.normalized_name = normalize_identity(name)

    # -------------------------------------------------------------------------
    def delete(self, dataset_id: int) -> None:
        with self.database.transaction() as session:
            session.execute(delete(Dataset).where(Dataset.id == dataset_id))
