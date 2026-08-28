from __future__ import annotations

import time

from sqlalchemy import inspect, text
from sqlalchemy.engine import Connection
from sqlalchemy.exc import SQLAlchemyError

from shared.common.settings import DatabaseSettings, get_server_settings
from shared.common.utils.logger import logger
from shared.repositories.database.legacy_schema import BASELINE_TABLES
from shared.repositories.database.manager import DatabaseManager
from shared.repositories.database.migrator import (
    DatabaseMigrationError,
    MIGRATION_LOCK_KEY,
    MigrationLockTimeoutError,
    MigrationResult,
    migrate_database,
)
from shared.repositories.database.sql import (
    build_postgres_create_database_sql,
    build_postgres_database_exists_sql,
)

###############################################################################
def clone_settings_with_database(
    settings: DatabaseSettings,
    database_name: str,
) -> DatabaseSettings:
    return DatabaseSettings(
        embedded_database=False,
        engine=settings.engine,
        host=settings.host,
        port=settings.port,
        database_name=database_name,
        username=settings.username,
        password=settings.password,
        ssl=settings.ssl,
        ssl_ca=settings.ssl_ca,
        connect_timeout=settings.connect_timeout,
        insert_batch_size=settings.insert_batch_size,
        sqlite_path=settings.sqlite_path,
    )

###############################################################################
def _validate_postgres_settings(settings: DatabaseSettings) -> None:
    if not settings.host or not settings.username or not settings.database_name:
        raise ValueError("PostgreSQL host, database name, and username are required.")

###############################################################################
def _acquire_creation_lock(connection: Connection, timeout_seconds: int) -> None:
    deadline = time.monotonic() + max(1, timeout_seconds)
    while True:
        locked = connection.execute(
            text("SELECT pg_try_advisory_lock(:lock_key)"),
            {"lock_key": MIGRATION_LOCK_KEY},
        ).scalar()
        if locked:
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise MigrationLockTimeoutError(
                f"Timed out after {timeout_seconds}s waiting for the PostgreSQL database lock."
            )
        time.sleep(min(0.1, remaining))

###############################################################################
def _ensure_postgres_database_exists(settings: DatabaseSettings) -> None:
    _validate_postgres_settings(settings)
    admin_manager: DatabaseManager | None = None
    try:
        admin_settings = clone_settings_with_database(settings, "postgres")
        admin_manager = DatabaseManager(admin_settings)
        admin_engine = admin_manager.engine.execution_options(
            isolation_level="AUTOCOMMIT"
        )
        with admin_engine.connect() as connection:
            _acquire_creation_lock(connection, settings.connect_timeout)
            try:
                exists = connection.execute(
                    build_postgres_database_exists_sql(),
                    {"name": settings.database_name},
                ).scalar()
                if not exists:
                    logger.info(
                        "Creating PostgreSQL database %s before running Alembic.",
                        settings.database_name,
                    )
                    connection.execute(
                        build_postgres_create_database_sql(settings.database_name)
                    )
            finally:
                connection.execute(
                    text("SELECT pg_advisory_unlock(:lock_key)"),
                    {"lock_key": MIGRATION_LOCK_KEY},
                )
    except DatabaseMigrationError:
        raise
    except SQLAlchemyError as exc:
        detail = str(getattr(exc, "orig", exc)).lower()
        if "permission denied" in detail and "database" in detail:
            raise DatabaseMigrationError(
                "The configured PostgreSQL role lacks CREATEDB permission; "
                "grant CREATEDB or pre-create the target database."
            ) from exc
        raise DatabaseMigrationError(
            "PostgreSQL database creation failed; verify the configured host, "
            "credentials, and CREATEDB permission."
        ) from exc
    except ValueError as exc:
        raise DatabaseMigrationError(
            "PostgreSQL database creation failed; verify the configured host, "
            "credentials, and CREATEDB permission."
        ) from exc
    finally:
        if admin_manager is not None:
            admin_manager.dispose()

###############################################################################
def verify_postgres_database(settings: DatabaseSettings) -> None:
    """Perform a non-mutating connectivity/schema probe for diagnostics."""

    _validate_postgres_settings(settings)
    manager: DatabaseManager | None = None
    try:
        manager = DatabaseManager(settings)
        with manager.engine.connect() as connection:
            connection.execute(text("SELECT 1"))
            tables = set(inspect(connection).get_table_names())
        if not BASELINE_TABLES.issubset(tables):
            raise RuntimeError(
                "PostgreSQL schema is incomplete; run database initialization."
            )
    except (SQLAlchemyError, ValueError) as exc:
        logger.error("PostgreSQL readiness probe failed (%s).", type(exc).__name__)
        raise RuntimeError(
            "PostgreSQL is unavailable or not initialized; inspect database settings."
        ) from None
    finally:
        if manager is not None:
            manager.dispose()

###############################################################################
def initialize_sqlite_database(settings: DatabaseSettings) -> MigrationResult:
    return migrate_database(settings)

###############################################################################
def initialize_postgres_database(settings: DatabaseSettings) -> MigrationResult:
    _ensure_postgres_database_exists(settings)
    return migrate_database(settings)

###############################################################################
def prepare_database_for_startup(
    settings: DatabaseSettings | None = None,
) -> MigrationResult:
    database_settings = settings or get_server_settings().database
    if database_settings.embedded_database:
        return initialize_sqlite_database(database_settings)
    return initialize_postgres_database(database_settings)

###############################################################################
def run_database_initialization(
    settings: DatabaseSettings | None = None,
) -> MigrationResult:
    return prepare_database_for_startup(settings)

###############################################################################
def initialize_database(settings: DatabaseSettings | None = None) -> MigrationResult:
    try:
        return run_database_initialization(settings)
    except DatabaseMigrationError as exc:
        logger.error("Database initialization failed: %s", exc)
        raise RuntimeError(str(exc)) from None
    except (SQLAlchemyError, OSError, ValueError) as exc:
        logger.error("Database initialization failed (%s).", type(exc).__name__)
        raise RuntimeError(
            "Database initialization failed; verify the configured database is reachable."
        ) from None


__all__ = [
    "clone_settings_with_database",
    "initialize_database",
    "initialize_postgres_database",
    "initialize_sqlite_database",
    "prepare_database_for_startup",
    "run_database_initialization",
    "verify_postgres_database",
]
