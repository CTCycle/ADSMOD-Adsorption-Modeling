# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations
import threading
import pytest
import sqlalchemy
from concurrent.futures import ThreadPoolExecutor
from sqlalchemy import event, select
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.exc import IntegrityError, InterfaceError
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool
from server.repositories.database.bulk import upsert_records
from server.repositories.database.upsert import resolve_conflict_columns
from server.repositories.schemas.models import Adsorbate, Base
from collections.abc import Iterator
from server.configurations.settings import DatabaseConfig
from server.repositories.snapshots import SnapshotStore
from server.repositories.database.manager import DatabaseManager


# Material upsert


###############################################################################
def configure_sqlite_connection(dbapi_connection, connection_record) -> None:  # type: ignore[no-untyped-def]
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA busy_timeout=30000")
    finally:
        cursor.close()


###############################################################################
def run_concurrent_upsert(
    barrier: threading.Barrier,
    engine: sqlalchemy.Engine,
    payload: dict[str, object],
    conflict_columns: list[str],
) -> None:
    barrier.wait()
    run_upsert_with_retry(engine, payload, conflict_columns)


###############################################################################
def run_upsert_with_retry(
    engine: sqlalchemy.Engine,
    payload: dict[str, object],
    conflict_columns: list[str],
    retries: int = 2,
) -> None:
    attempts = 0
    while True:
        try:
            upsert_adsorbate(engine, payload, conflict_columns)
            return
        except InterfaceError:
            attempts += 1
            if attempts > retries:
                raise


###############################################################################
def build_sqlite_engine() -> sqlalchemy.Engine:
    engine = sqlalchemy.create_engine(
        "sqlite://",
        future=True,
        connect_args={"check_same_thread": False, "timeout": 30},
        poolclass=StaticPool,
    )
    event.listen(engine, "connect", configure_sqlite_connection)
    Base.metadata.create_all(engine)
    return engine


###############################################################################
def upsert_adsorbate(
    engine: sqlalchemy.Engine,
    payload: dict[str, object],
    conflict_columns: list[str],
) -> None:
    with Session(engine) as session:
        upsert_records(session, Adsorbate.__table__, [payload], conflict_columns)
        session.commit()


###############################################################################
def test_wrong_conflict_target_can_raise_duplicate_adsorbate_identity() -> None:
    engine = build_sqlite_engine()

    first = {
        "key": "name:0001",
        "name": "methane",
        "normalized_name": "methane",
        "inchi_key": None,
        "formula": "CH4",
    }
    second = {
        "key": "name:0001",
        "name": "methane",
        "normalized_name": "methane",
        "inchi_key": None,
        "formula": "CH4-updated",
    }

    upsert_adsorbate(engine, first, ["inchi_key"])
    with Session(engine) as session:
        statement = (
            insert(Adsorbate.__table__)
            .values([second])
            .on_conflict_do_update(
                index_elements=["inchi_key"],
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
    assert conflict_columns == ["key"]

    first = {
        "key": "name:0002",
        "name": "ethane",
        "normalized_name": "ethane",
        "inchi_key": None,
        "formula": "C2H6",
    }
    second = {
        "key": "name:0002",
        "name": "ethane",
        "normalized_name": "ethane",
        "inchi_key": None,
        "formula": "C2H6-updated",
        "molar_mass_g_mol": 30.07,
    }

    upsert_adsorbate(engine, first, conflict_columns)
    upsert_adsorbate(engine, second, conflict_columns)

    with Session(engine) as session:
        rows = (
            session.execute(select(Adsorbate).where(Adsorbate.key == "name:0002"))
            .scalars()
            .all()
        )
        assert len(rows) == 1
        assert rows[0].id is not None
        assert rows[0].formula == "C2H6-updated"
        assert rows[0].molar_mass_g_mol == 30.07

    engine.dispose()


###############################################################################
def test_concurrent_upsert_keeps_single_adsorbate_identity() -> None:
    engine = build_sqlite_engine()
    conflict_columns = resolve_conflict_columns(Adsorbate.__table__)
    barrier = threading.Barrier(2)

    payload_a = {
        "key": "name:0003",
        "name": "propane",
        "normalized_name": "propane",
        "inchi_key": None,
        "formula": "C3H8-a",
    }
    payload_b = {
        "key": "name:0003",
        "name": "propane",
        "normalized_name": "propane",
        "inchi_key": None,
        "formula": "C3H8-b",
    }

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                run_concurrent_upsert,
                barrier,
                engine,
                payload_a,
                conflict_columns,
            ),
            executor.submit(
                run_concurrent_upsert,
                barrier,
                engine,
                payload_b,
                conflict_columns,
            ),
        ]
        for future in futures:
            future.result()

    with Session(engine) as session:
        rows = (
            session.execute(select(Adsorbate).where(Adsorbate.key == "name:0003"))
            .scalars()
            .all()
        )
        assert len(rows) == 1
        assert rows[0].id is not None
        assert rows[0].formula in {"C3H8-a", "C3H8-b"}

    engine.dispose()


# Snapshots


###############################################################################
@pytest.fixture
def database() -> Iterator[DatabaseManager]:
    manager = DatabaseManager(
        DatabaseConfig(
            embedded_database=True,
            connect_timeout=5,
            sqlite_path=":memory:",
        )
    )
    Base.metadata.create_all(manager.engine)
    try:
        yield manager
    finally:
        manager.dispose()


###############################################################################
def _rows() -> list[dict[str, int | str]]:
    return [{"id": index, "value": f"value-{index}"} for index in range(7)]


###############################################################################
@pytest.mark.parametrize(
    ("page_number", "expected_ids"),
    [
        pytest.param(1, (0, 1, 2), id="first-page"),
        pytest.param(2, (3, 4, 5), id="middle-page"),
        pytest.param(3, (6,), id="last-page"),
        pytest.param(4, (), id="offset-beyond-end"),
    ],
)
def test_get_page_returns_ordered_slice_and_total(
    database: DatabaseManager,
    page_number: int,
    expected_ids: tuple[int, ...],
) -> None:
    store = SnapshotStore(database)
    rows = _rows()
    record = store.create(rows)

    result = store.get_page(record.snapshot_id, page_number, 3)

    assert result.snapshot_id == record.snapshot_id
    assert result.content_hash == record.content_hash
    assert result.page == page_number
    assert result.page_size == 3
    assert result.total_rows == len(rows)
    assert tuple(row["id"] for row in result.rows) == expected_ids
    assert result.rows == tuple(rows[(page_number - 1) * 3 : page_number * 3])


###############################################################################
def test_create_reuses_existing_snapshot_for_same_content(
    database: DatabaseManager,
) -> None:
    store = SnapshotStore(database)
    rows = _rows()

    first = store.create(rows, metadata={"attempt": 1})
    second = store.create(rows, metadata={"attempt": 2})

    assert second.snapshot_id == first.snapshot_id
    assert second.content_hash == first.content_hash
    assert second.created_at == first.created_at
    assert second.rows == first.rows


###############################################################################
def test_get_page_uses_ordered_bounded_row_query(
    database: DatabaseManager,
) -> None:
    store = SnapshotStore(database)
    record = store.create(_rows())
    statements: list[str] = []

    def capture_select(
        conn,
        cursor,
        statement,
        parameters,
        context,
        executemany,  # type: ignore[no-untyped-def]
    ) -> None:
        del conn, cursor, parameters, context, executemany
        if statement.lstrip().upper().startswith("SELECT"):
            statements.append(statement)

    event.listen(database.engine, "before_cursor_execute", capture_select)
    try:
        result = store.get_page(record.snapshot_id, 2, 3)
    finally:
        event.remove(database.engine, "before_cursor_execute", capture_select)

    row_queries = [
        statement
        for statement in statements
        if "TRAINING_SNAPSHOT_ROWS" in statement.upper()
    ]
    assert len(statements) == 2
    assert len(row_queries) == 1
    row_query = " ".join(row_queries[0].upper().split())
    assert "WHERE TRAINING_SNAPSHOT_ROWS.SNAPSHOT_ID" in row_query
    assert "ORDER BY TRAINING_SNAPSHOT_ROWS.ROW_INDEX" in row_query
    assert "LIMIT" in row_query
    assert "OFFSET" in row_query
    assert tuple(row["id"] for row in result.rows) == (3, 4, 5)


###############################################################################
def test_get_page_preserves_validation_and_missing_snapshot_errors(
    database: DatabaseManager,
) -> None:
    store = SnapshotStore(database)

    with pytest.raises(ValueError, match="page must be >= 1"):
        store.get_page("missing", 0, 3)
    with pytest.raises(ValueError, match="page_size must be between 1 and 1000"):
        store.get_page("missing", 1, 0)
    with pytest.raises(KeyError):
        store.get_page("missing", 1, 3)
