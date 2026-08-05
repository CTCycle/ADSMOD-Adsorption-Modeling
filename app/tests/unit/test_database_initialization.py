from __future__ import annotations

import hashlib
from pathlib import Path
import sqlite3

import pytest
from sqlalchemy.exc import SQLAlchemyError

from shared.common.settings import DatabaseSettings
from shared.repositories.database import initializer


###############################################################################
def sqlite_settings(path: Path) -> DatabaseSettings:
    return DatabaseSettings(
        embedded_database=True,
        engine=None,
        host=None,
        port=None,
        database_name=None,
        username=None,
        password=None,
        ssl=False,
        ssl_ca=None,
        connect_timeout=30,
        insert_batch_size=100,
        sqlite_path=str(path),
    )


###############################################################################
def postgres_settings() -> DatabaseSettings:
    return DatabaseSettings(
        embedded_database=False,
        engine="postgres",
        host="127.0.0.1",
        port=5432,
        database_name="adsmod_test_lifecycle",
        username="postgres",
        password="secret",
        ssl=False,
        ssl_ca=None,
        connect_timeout=1,
        insert_batch_size=100,
    )


###############################################################################
def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


###############################################################################
def test_sqlite_startup_creates_missing_database_once(tmp_path: Path) -> None:
    database_path = tmp_path / "database.db"
    settings = sqlite_settings(database_path)

    initializer.prepare_database_for_startup(settings)

    assert database_path.is_file()
    with sqlite3.connect(database_path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    assert "datasets" in tables

    digest_before = file_sha256(database_path)
    mtime_before = database_path.stat().st_mtime_ns
    initializer.initialize_database(settings)

    assert file_sha256(database_path) == digest_before
    assert database_path.stat().st_mtime_ns == mtime_before


###############################################################################
def test_existing_sqlite_file_is_not_initialized_or_repaired(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database_path = tmp_path / "existing.db"
    database_path.touch()
    settings = sqlite_settings(database_path)

    def fail_if_constructed(*args: object, **kwargs: object) -> None:
        raise AssertionError("existing SQLite files must not construct an initializer")

    monkeypatch.setattr(initializer.DatabaseManager, "__init__", fail_if_constructed)

    initializer.prepare_database_for_startup(settings)
    initializer.initialize_database(settings)

    assert database_path.stat().st_size == 0


###############################################################################
def test_postgres_startup_uses_readiness_check_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[DatabaseSettings] = []
    monkeypatch.setattr(
        initializer,
        "verify_postgres_database",
        lambda settings: calls.append(settings),
    )
    monkeypatch.setattr(
        initializer,
        "initialize_postgres_database",
        lambda settings: pytest.fail("startup must not initialize PostgreSQL"),
    )

    settings = postgres_settings()
    initializer.prepare_database_for_startup(settings)

    assert calls == [settings]


###############################################################################
def test_postgres_readiness_checks_connection_and_required_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    statements: list[str] = []

    ###############################################################################
    class Result:

        # -------------------------------------------------------------------------
        def scalar(self) -> int:
            return 1

    ###############################################################################
    class Connection:

        # -------------------------------------------------------------------------
        def __enter__(self) -> "Connection":
            return self

        # -------------------------------------------------------------------------
        def __exit__(self, *args: object) -> None:
            return None

        # -------------------------------------------------------------------------
        def execute(self, statement: object) -> Result:
            statements.append(str(statement))
            return Result()

    ###############################################################################
    class Engine:

        # -------------------------------------------------------------------------
        def connect(self) -> Connection:
            return Connection()

    ###############################################################################
    class Manager:

        # -------------------------------------------------------------------------
        def __init__(self, settings: DatabaseSettings) -> None:
            self.engine = Engine()

        # -------------------------------------------------------------------------
        def dispose(self) -> None:
            return None

    monkeypatch.setattr(initializer, "DatabaseManager", Manager)

    initializer.verify_postgres_database(postgres_settings())

    assert any(statement.strip() == "SELECT 1" for statement in statements)
    assert any("information_schema.tables" in statement for statement in statements)


###############################################################################
def test_postgres_readiness_rejects_missing_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = iter([None])

    ###############################################################################
    class Result:

        # -------------------------------------------------------------------------
        def scalar(self) -> int | None:
            return next(values)

    ###############################################################################
    class Connection:

        # -------------------------------------------------------------------------
        def __enter__(self) -> "Connection":
            return self

        # -------------------------------------------------------------------------
        def __exit__(self, *args: object) -> None:
            return None

        # -------------------------------------------------------------------------
        def execute(self, statement: object) -> Result:
            return Result()

    ###############################################################################
    class Engine:

        # -------------------------------------------------------------------------
        def connect(self) -> Connection:
            return Connection()

    ###############################################################################
    class Manager:

        # -------------------------------------------------------------------------
        def __init__(self, settings: DatabaseSettings) -> None:
            self.engine = Engine()

        # -------------------------------------------------------------------------
        def dispose(self) -> None:
            return None

    monkeypatch.setattr(initializer, "DatabaseManager", Manager)

    with pytest.raises(RuntimeError, match="not initialized"):
        initializer.verify_postgres_database(postgres_settings())


###############################################################################
def test_explicit_postgres_initialization_creates_database_and_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    statements: list[str] = []
    database_names: list[str | None] = []

    ###############################################################################
    class Result:

        # -------------------------------------------------------------------------
        def scalar(self) -> int:
            return 0

    ###############################################################################
    class Connection:

        # -------------------------------------------------------------------------
        def __enter__(self) -> "Connection":
            return self

        # -------------------------------------------------------------------------
        def __exit__(self, *args: object) -> None:
            return None

        # -------------------------------------------------------------------------
        def execute(self, statement: object, params: object = None) -> Result:
            statements.append(str(statement))
            return Result()

    ###############################################################################
    class Engine:

        # -------------------------------------------------------------------------
        def execution_options(self, **kwargs: object) -> "Engine":
            assert kwargs == {"isolation_level": "AUTOCOMMIT"}
            return self

        # -------------------------------------------------------------------------
        def connect(self) -> Connection:
            return Connection()

    ###############################################################################
    class Manager:

        # -------------------------------------------------------------------------
        def __init__(self, settings: DatabaseSettings) -> None:
            database_names.append(settings.database_name)
            self.engine = Engine()

        # -------------------------------------------------------------------------
        def dispose(self) -> None:
            return None

    monkeypatch.setattr(initializer, "DatabaseManager", Manager)
    monkeypatch.setattr(
        initializer.Base.metadata,
        "create_all",
        lambda engine: statements.append("CREATE SCHEMA"),
    )

    initializer.initialize_postgres_database(postgres_settings())

    assert database_names == ["postgres", "adsmod_test_lifecycle"]
    assert any("pg_database" in statement for statement in statements)
    assert any("CREATE DATABASE" in statement for statement in statements)


###############################################################################
def test_postgres_startup_failure_does_not_expose_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    ###############################################################################
    class FailingManager:

        # -------------------------------------------------------------------------
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise SQLAlchemyError("postgresql://postgres:secret@host/database")

    monkeypatch.setattr(initializer, "DatabaseManager", FailingManager)

    with pytest.raises(RuntimeError) as error:
        initializer.verify_postgres_database(postgres_settings())

    assert "secret" not in str(error.value)
    assert "not initialized" in str(error.value)


###############################################################################
def test_database_initialization_failure_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        initializer,
        "run_database_initialization",
        lambda settings=None: (_ for _ in ()).throw(
            SQLAlchemyError("postgresql://postgres:secret@host/database")
        ),
    )

    with pytest.raises(RuntimeError) as error:
        initializer.initialize_database(sqlite_settings(Path("unused.db")))

    assert "secret" not in str(error.value)
    assert "verify" in str(error.value)
