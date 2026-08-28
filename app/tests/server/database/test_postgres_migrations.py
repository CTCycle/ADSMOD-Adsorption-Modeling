from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
from uuid import uuid4

import pytest
from sqlalchemy import inspect, text
from sqlalchemy.exc import SQLAlchemyError

from shared.common.settings import DatabaseSettings
from shared.repositories.database import initializer, migrator
from shared.repositories.database.initializer import clone_settings_with_database
from shared.repositories.database.legacy_schema import (
    BASELINE_TABLES,
    schema_matches_baseline,
)
from shared.repositories.database.manager import DatabaseManager
from shared.repositories.schemas.models import Base, Dataset

###############################################################################
def _postgres_settings(database_name: str) -> DatabaseSettings | None:
    host = os.getenv("DATABASE_HOST")
    if (
        not host
        or os.getenv("DATABASE_EMBEDDED", "true").strip().lower() != "false"
    ):
        return None
    return DatabaseSettings(
        embedded_database=False,
        engine=os.getenv("DATABASE_ENGINE", "postgres"),
        host=host,
        port=int(os.getenv("DATABASE_PORT", "5432")),
        database_name=database_name,
        username=os.getenv("DATABASE_USERNAME", "postgres"),
        password=os.getenv("DATABASE_PASSWORD", "postgres"),
        ssl=False,
        ssl_ca=None,
        connect_timeout=int(os.getenv("DATABASE_CONNECT_TIMEOUT", "10")),
        insert_batch_size=100,
        sqlite_path=None,
    )

###############################################################################
@pytest.fixture
def postgres_settings() -> DatabaseSettings:
    settings = _postgres_settings(f"adsmod_alembic_{uuid4().hex[:12]}")
    if settings is None:
        pytest.skip("PostgreSQL migration tests require DATABASE_HOST and DATABASE_EMBEDDED=false")
    assert settings is not None
    try:
        yield settings
    finally:
        admin = DatabaseManager(clone_settings_with_database(settings, "postgres"))
        try:
            engine = admin.engine.execution_options(isolation_level="AUTOCOMMIT")
            with engine.connect() as connection:
                safe_name = settings.database_name.replace('"', '""')
                connection.execute(text(f'DROP DATABASE IF EXISTS "{safe_name}"'))
        finally:
            admin.dispose()

###############################################################################
def test_postgres_fresh_and_repeated_initialization(
    postgres_settings: DatabaseSettings,
) -> None:
    first = initializer.initialize_postgres_database(postgres_settings)
    second = initializer.initialize_postgres_database(postgres_settings)

    assert first.after == (first.head,)
    assert first.applied_migrations is True
    assert second.after == (first.head,)
    assert second.applied_migrations is False

    manager = DatabaseManager(postgres_settings)
    try:
        with manager.engine.connect() as connection:
            assert BASELINE_TABLES.issubset(set(inspect(connection).get_table_names()))
            assert schema_matches_baseline(inspect(connection))
    finally:
        manager.dispose()

###############################################################################
def test_postgres_legacy_adoption_preserves_data(
    postgres_settings: DatabaseSettings,
) -> None:
    initializer._ensure_postgres_database_exists(postgres_settings)
    manager = DatabaseManager(postgres_settings)
    try:
        Base.metadata.create_all(manager.engine)
        with manager.transaction() as session:
            session.add(Dataset(name="Legacy PostgreSQL", source="uploaded"))
        with manager.engine.connect() as connection:
            assert schema_matches_baseline(inspect(connection))
    finally:
        manager.dispose()

    result = initializer.initialize_postgres_database(postgres_settings)
    assert result.adopted_legacy_schema is True

    manager = DatabaseManager(postgres_settings)
    try:
        with manager.engine.connect() as connection:
            assert connection.execute(
                text("SELECT count(*) FROM datasets WHERE name = 'Legacy PostgreSQL'")
            ).scalar_one() == 1
    finally:
        manager.dispose()

###############################################################################
def test_postgres_advisory_lock_serializes_concurrent_startup(
    postgres_settings: DatabaseSettings,
) -> None:
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(initializer.initialize_postgres_database, [postgres_settings] * 2)
        )

    assert all(result.after == (result.head,) for result in results)
    assert sum(result.applied_migrations for result in results) == 1

###############################################################################
def test_postgres_migration_failure_rolls_back(
    postgres_settings: DatabaseSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_after_partial_ddl(
        config: object,
        connection: object,
        action: str,
        revision: str,
    ) -> None:
        del config, action, revision
        assert hasattr(connection, "execute")
        connection.execute(text("CREATE TABLE migration_transient (id integer)"))
        raise SQLAlchemyError("simulated migration failure")

    monkeypatch.setattr(migrator, "_run_command", fail_after_partial_ddl)
    initializer._ensure_postgres_database_exists(postgres_settings)
    with pytest.raises(migrator.DatabaseMigrationError):
        migrator.migrate_database(postgres_settings)

    manager = DatabaseManager(postgres_settings)
    try:
        with manager.engine.connect() as connection:
            tables = set(inspect(connection).get_table_names())
            assert "migration_transient" not in tables
            assert "alembic_version" not in tables
    finally:
        manager.dispose()
