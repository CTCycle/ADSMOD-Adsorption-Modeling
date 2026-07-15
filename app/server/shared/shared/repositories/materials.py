from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from sqlalchemy import select

from shared.repositories.database.bulk import upsert_records
from shared.repositories.database.manager import DatabaseManager
from shared.repositories.schemas.models import Adsorbate, Adsorbent
from shared.repositories.schemas.types import normalize_identity


class MaterialRepository:
    def __init__(self, database: DatabaseManager) -> None:
        self.database = database

    def upsert_adsorbates(self, records: Iterable[dict[str, Any]]) -> int:
        rows = [dict(row, normalized_name=normalize_identity(str(row.get("name", row["adsorbate_key"])))) for row in records]
        with self.database.transaction() as session:
            return upsert_records(session, Adsorbate.__table__, rows, ["adsorbate_key"])

    def upsert_adsorbents(self, records: Iterable[dict[str, Any]]) -> int:
        rows = [dict(row, normalized_name=normalize_identity(str(row.get("name", row["adsorbent_key"])))) for row in records]
        with self.database.transaction() as session:
            return upsert_records(session, Adsorbent.__table__, rows, ["adsorbent_key"])

    def adsorbate_ids(self, keys: Iterable[str]) -> dict[str, int]:
        with self.database.session_factory() as session:
            rows = session.execute(select(Adsorbate.adsorbate_key, Adsorbate.id).where(Adsorbate.adsorbate_key.in_(list(keys)))).all()
            return {key: identifier for key, identifier in rows}

    def adsorbent_ids(self, keys: Iterable[str]) -> dict[str, int]:
        with self.database.session_factory() as session:
            rows = session.execute(select(Adsorbent.adsorbent_key, Adsorbent.id).where(Adsorbent.adsorbent_key.in_(list(keys)))).all()
            return {key: identifier for key, identifier in rows}
