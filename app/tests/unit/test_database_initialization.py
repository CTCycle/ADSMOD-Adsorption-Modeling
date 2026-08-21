from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sqlite3

import pytest
from sqlalchemy import inspect
from sqlalchemy.exc import SQLAlchemyError

from shared.common.settings import DatabaseSettings
from shared.repositories.database import initializer
from shared.repositories.database.legacy_schema import (
    BASELINE_SCHEMA_HASH,
    inspector_signature,
    schema_hash,
)
from shared.repositories.database.manager import DatabaseManager
from shared.repositories.database.migrator import (
    DatabaseMigrationError,
    LegacySchemaMismatchError,
    MigrationLockTimeoutError,
    build_alembic_config,
    migrate_database,
)
from shared.repositories.schemas.models import Base, Dataset


###############################################################################
def sqlite_settings(path: Path, *, timeout: int = 5) -> DatabaseSettings:
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
        connect_timeout=timeout,
        insert_batch_size=100,
        sqlite_path=str(path),
    )


###############################################################################
def table_names(path: Path) -> set[str]:
    connection = sqlite3.connect(path)
    try:
        return {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    finally:
        connection.close()


###############################################################################
def test_sqlite_missing_database_runs_baseline_and_is_idempotent(
    tmp_path: Path,
) -> None:
    path = tmp_path / "database.db"
    settings = sqlite_settings(path)

    first = initializer.prepare_database_for_startup(settings)
    second = initializer.initialize_database(settings)

    assert first.applied_migrations is True
    assert second.applied_migrations is False
    assert "alembic_version" in table_names(path)
    assert len(table_names(path) & {table.name for table in Base.metadata.tables.values()}) == 12


###############################################################################
def test_sqlite_empty_existing_file_is_initialized(tmp_path: Path) -> None:
    path = tmp_path / "empty.db"
    path.touch()

    result = migrate_database(sqlite_settings(path))

    assert result.after == (result.head,)
    assert table_names(path) >= {"alembic_version", "datasets"}


###############################################################################
def test_legacy_schema_is_validated_stamped_and_data_is_preserved(
    tmp_path: Path,
) -> None:
    path = tmp_path / "legacy.db"
    settings = sqlite_settings(path)
    manager = DatabaseManager(settings)
    try:
        Base.metadata.create_all(manager.engine)  # Simulate the pre-Alembic release.
        with manager.transaction() as session:
            session.add(
                Dataset(
                    name="Legacy Water",
                    source="uploaded",
                    description="retain me",
                )
            )
    finally:
        manager.dispose()

    result = migrate_database(settings)

    assert result.adopted_legacy_schema is True
    check = sqlite3.connect(path)
    try:
        assert check.execute("SELECT name FROM datasets").fetchone() == ("Legacy Water",)
        assert check.execute("SELECT version_num FROM alembic_version").fetchone() == (
            result.head,
        )
    finally:
        check.close()


###############################################################################
def test_legacy_schema_hash_matches_reviewed_baseline(tmp_path: Path) -> None:
    path = tmp_path / "legacy.db"
    manager = DatabaseManager(sqlite_settings(path))
    try:
        Base.metadata.create_all(manager.engine)
        with manager.engine.connect() as connection:
            assert schema_hash(inspector_signature(inspect(connection))) == BASELINE_SCHEMA_HASH
    finally:
        manager.dispose()


###############################################################################
def test_unversioned_schema_mismatch_is_not_modified(tmp_path: Path) -> None:
    path = tmp_path / "mismatch.db"
    manager = DatabaseManager(sqlite_settings(path))
    try:
        Base.metadata.create_all(manager.engine)
    finally:
        manager.dispose()
    with sqlite3.connect(path) as connection:
        connection.execute("ALTER TABLE datasets ADD COLUMN legacy_extra TEXT")

    with pytest.raises(LegacySchemaMismatchError):
        migrate_database(sqlite_settings(path))

    assert "alembic_version" not in table_names(path)
    with sqlite3.connect(path) as connection:
        assert "legacy_extra" in {
            row[1] for row in connection.execute("PRAGMA table_info(datasets)")
        }


###############################################################################
def test_unversioned_unrelated_table_is_not_modified(tmp_path: Path) -> None:
    path = tmp_path / "unrelated.db"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE unrelated (id INTEGER PRIMARY KEY)")

    with pytest.raises(LegacySchemaMismatchError):
        migrate_database(sqlite_settings(path))

    assert table_names(path) == {"unrelated"}


###############################################################################
def test_empty_version_table_with_application_tables_fails_safely(tmp_path: Path) -> None:
    path = tmp_path / "interrupted.db"
    manager = DatabaseManager(sqlite_settings(path))
    try:
        Base.metadata.create_all(manager.engine)
    finally:
        manager.dispose()
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)")
        connection.commit()

    with pytest.raises(RuntimeError, match="empty alembic_version"):
        migrate_database(sqlite_settings(path))


###############################################################################
def test_failed_migration_rolls_back_schema(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "rollback.db"
    settings = sqlite_settings(path)
    config = build_alembic_config()

    def fail_after_ddl(config, connection, action, revision):  # type: ignore[no-untyped-def]
        connection.exec_driver_sql("CREATE TABLE transient_failure (id INTEGER)")
        raise SQLAlchemyError("synthetic migration failure")

    monkeypatch.setattr(
        "shared.repositories.database.migrator._run_command",
        fail_after_ddl,
    )

    with pytest.raises(SQLAlchemyError, match="synthetic"):
        from shared.repositories.database.migrator import migrate_engine

        manager = DatabaseManager(settings)
        try:
            migrate_engine(manager.engine, settings, config=config)
        finally:
            manager.dispose()

    assert "transient_failure" not in table_names(path)
    assert "alembic_version" not in table_names(path)


###############################################################################
def test_concurrent_sqlite_startup_serializes_migrations(tmp_path: Path) -> None:
    path = tmp_path / "concurrent.db"

    def start_once():  # type: ignore[no-untyped-def]
        return migrate_database(sqlite_settings(path))

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: start_once(), range(2)))

    assert all(result.after == (result.head,) for result in results)
    assert table_names(path) >= {"alembic_version", "datasets"}


###############################################################################
def test_sqlite_migration_lock_timeout_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "locked.db"
    blocker = sqlite3.connect(path, timeout=5)
    blocker.execute("BEGIN IMMEDIATE")
    try:
        with pytest.raises(MigrationLockTimeoutError):
            migrate_database(sqlite_settings(path, timeout=1))
    finally:
        blocker.rollback()
        blocker.close()


###############################################################################
def test_database_initialization_sanitizes_sqlalchemy_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        initializer,
        "run_database_initialization",
        lambda settings=None: (_ for _ in ()).throw(
            SQLAlchemyError("postgresql://postgres:secret@host/database")
        ),
    )

    with pytest.raises(RuntimeError) as error:
        initializer.initialize_database(sqlite_settings(tmp_path / "unused.db"))

    assert "secret" not in str(error.value)
    assert "verify" in str(error.value)


###############################################################################
def test_postgres_creation_permission_error_is_actionable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = DatabaseSettings(
        embedded_database=False,
        engine="postgres",
        host="localhost",
        port=5432,
        database_name="adsmod_permission_test",
        username="app",
        password="secret",
        ssl=False,
        ssl_ca=None,
        connect_timeout=1,
        insert_batch_size=100,
        sqlite_path=None,
    )

    def fail_to_connect(_settings: DatabaseSettings) -> DatabaseManager:
        raise SQLAlchemyError("permission denied to create database")

    monkeypatch.setattr(initializer, "DatabaseManager", fail_to_connect)
    with pytest.raises(DatabaseMigrationError, match="CREATEDB"):
        initializer._ensure_postgres_database_exists(settings)
