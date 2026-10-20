# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from sqlalchemy import select

from server.repositories.database.bulk import upsert_records
from server.repositories.database.manager import DatabaseManager
from server.repositories.schemas.models import Adsorbate, Adsorbent
from server.repositories.schemas.types import normalize_identity

###############################################################################
class MaterialRepository:

    # -------------------------------------------------------------------------
    def __init__(self, database: DatabaseManager) -> None:
        self.database = database

    # -------------------------------------------------------------------------
    def upsert_adsorbates(self, records: Iterable[dict[str, Any]]) -> int:
        columns = Adsorbate.__table__.c
        rows = [
            {key: value for key, value in row.items() if key in columns}
            | {"normalized_name": normalize_identity(str(row.get("name", row["key"])))}
            for row in records
        ]
        with self.database.transaction() as session:
            return upsert_records(
                session, Adsorbate.__table__, rows, ["key"], keep_existing_on_null=True
            )

    # -------------------------------------------------------------------------
    def upsert_adsorbents(self, records: Iterable[dict[str, Any]]) -> int:
        columns = Adsorbent.__table__.c
        rows = [
            {key: value for key, value in row.items() if key in columns}
            | {"normalized_name": normalize_identity(str(row.get("name", row["key"])))}
            for row in records
        ]
        with self.database.transaction() as session:
            return upsert_records(
                session, Adsorbent.__table__, rows, ["key"], keep_existing_on_null=True
            )

    # -------------------------------------------------------------------------
    def adsorbate_ids(self, keys: Iterable[str]) -> dict[str, int]:
        with self.database.session_factory() as session:
            rows = session.execute(
                select(Adsorbate.key, Adsorbate.id).where(Adsorbate.key.in_(list(keys)))
            ).all()
            return {key: identifier for key, identifier in rows}

    # -------------------------------------------------------------------------
    def adsorbent_ids(self, keys: Iterable[str]) -> dict[str, int]:
        with self.database.session_factory() as session:
            rows = session.execute(
                select(Adsorbent.key, Adsorbent.id).where(Adsorbent.key.in_(list(keys)))
            ).all()
            return {key: identifier for key, identifier in rows}
