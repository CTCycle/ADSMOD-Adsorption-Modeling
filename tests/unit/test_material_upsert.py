from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
import sqlalchemy
from sqlalchemy import event, select
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from ADSMOD.server.repositories.database.upsert import resolve_conflict_columns
from ADSMOD.server.repositories.schemas.models import Adsorbate, Base


###############################################################################
def build_sqlite_engine() -> sqlalchemy.Engine:
    engine = sqlalchemy.create_engine(
        "sqlite://",
        future=True,
        connect_args={"check_same_thread": False, "timeout": 30},
        poolclass=StaticPool,
    )

    @event.listens_for(engine, "connect")
    def configure_connection(dbapi_connection, connection_record) -> None:  # type: ignore[no-untyped-def]
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA busy_timeout=30000")
        finally:
            cursor.close()

    Base.metadata.create_all(engine)
    return engine


# -----------------------------------------------------------------------------
def upsert_adsorbate(
    engine: sqlalchemy.Engine,
    payload: dict[str, object],
    conflict_columns: list[str],
) -> None:
    with Session(engine) as session:
        statement = insert(Adsorbate.__table__).values([payload])
        update_columns = {
            key: getattr(statement.excluded, key)
            for key in payload
            if key not in conflict_columns
        }
        statement = statement.on_conflict_do_update(
            index_elements=conflict_columns,
            set_=update_columns,
        )
        session.execute(statement)
        session.commit()


###############################################################################
def test_wrong_conflict_target_can_raise_duplicate_adsorbate_key() -> None:
    engine = build_sqlite_engine()

    first = {
        "adsorbate_key": "name:0001",
        "InChIKey": None,
        "name": "methane",
        "formula": "CH4",
    }
    second = {
        "adsorbate_key": "name:0001",
        "InChIKey": None,
        "name": "methane",
        "formula": "CH4-updated",
    }

    upsert_adsorbate(engine, first, ["InChIKey"])
    with Session(engine) as session:
        statement = (
            insert(Adsorbate.__table__)
            .values([second])
            .on_conflict_do_update(
                index_elements=["InChIKey"],
                set_={"formula": "CH4-updated"},
            )
        )
        with pytest.raises(IntegrityError):
            session.execute(statement)
            session.commit()

    engine.dispose()


###############################################################################
def test_retry_upsert_uses_single_adsorbate_row_id() -> None:
    engine = build_sqlite_engine()
    conflict_columns = resolve_conflict_columns(Adsorbate.__table__)

    first = {
        "adsorbate_key": "name:0002",
        "InChIKey": None,
        "name": "ethane",
        "formula": "C2H6",
    }
    second = {
        "adsorbate_key": "name:0002",
        "InChIKey": None,
        "name": "ethane",
        "formula": "C2H6-updated",
        "molecular_formula": "C2H6",
    }

    upsert_adsorbate(engine, first, conflict_columns)
    upsert_adsorbate(engine, second, conflict_columns)

    with Session(engine) as session:
        rows = (
            session.execute(
                select(Adsorbate).where(Adsorbate.adsorbate_key == "name:0002")
            )
            .scalars()
            .all()
        )
        assert len(rows) == 1
        assert rows[0].id is not None
        assert rows[0].formula == "C2H6-updated"
        assert rows[0].molecular_formula == "C2H6"

    engine.dispose()


###############################################################################
def test_concurrent_upsert_keeps_single_adsorbate_identity() -> None:
    engine = build_sqlite_engine()
    conflict_columns = resolve_conflict_columns(Adsorbate.__table__)
    barrier = threading.Barrier(2)

    payload_a = {
        "adsorbate_key": "name:0003",
        "InChIKey": None,
        "name": "propane",
        "formula": "C3H8-a",
    }
    payload_b = {
        "adsorbate_key": "name:0003",
        "InChIKey": None,
        "name": "propane",
        "formula": "C3H8-b",
    }

    def run_upsert(payload: dict[str, object]) -> None:
        barrier.wait()
        upsert_adsorbate(engine, payload, conflict_columns)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(run_upsert, payload_a),
            executor.submit(run_upsert, payload_b),
        ]
        for future in futures:
            future.result()

    with Session(engine) as session:
        rows = (
            session.execute(
                select(Adsorbate).where(Adsorbate.adsorbate_key == "name:0003")
            )
            .scalars()
            .all()
        )
        assert len(rows) == 1
        assert rows[0].id is not None
        assert rows[0].formula in {"C3H8-a", "C3H8-b"}

    engine.dispose()
