from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from shared.common.settings import DatabaseSettings, get_server_settings
from shared.common.utils.logger import logger
from shared.repositories.database.manager import DatabaseManager, resolve_sqlite_path
from shared.repositories.database.sql import (
    build_postgres_create_database_sql,
    build_postgres_database_exists_sql,
)
from shared.repositories.schemas.models import Base

POSTGRES_SCHEMA_READY_SQL = text(
    """
    SELECT 1
    FROM information_schema.tables
    WHERE table_schema = 'public' AND table_name = 'datasets'
    """
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
def initialize_sqlite_database(settings: DatabaseSettings) -> None:
    database_path = resolve_sqlite_path(settings)
    if database_path.is_file():
        logger.info("SQLite database already exists; skipping initialization.")
        return

    _create_schema(settings, "SQLite")

###############################################################################
def _create_schema(settings: DatabaseSettings, database_label: str) -> None:
    manager = DatabaseManager(settings)
    try:
        Base.metadata.create_all(manager.engine)
        logger.info("Initialized %s database schema.", database_label)
    finally:
        manager.dispose()

###############################################################################
def _validate_postgres_settings(settings: DatabaseSettings) -> None:
    if not settings.host or not settings.username or not settings.database_name:
        raise ValueError("PostgreSQL host, database name, and username are required.")

###############################################################################
def verify_postgres_database(settings: DatabaseSettings) -> None:
    _validate_postgres_settings(settings)
    manager: DatabaseManager | None = None
    try:
        manager = DatabaseManager(settings)
        with manager.engine.connect() as connection:
            connection.execute(text("SELECT 1"))
            schema_ready = connection.execute(POSTGRES_SCHEMA_READY_SQL).scalar()
        if not schema_ready:
            raise RuntimeError(
                "PostgreSQL schema is not initialized; run Initialize database."
            )
    except (SQLAlchemyError, ValueError) as exc:
        logger.error("PostgreSQL startup check failed (%s).", type(exc).__name__)
        raise RuntimeError(
            "PostgreSQL is unavailable or not initialized; run Initialize database."
        ) from None
    finally:
        if manager is not None:
            manager.dispose()

###############################################################################
def initialize_postgres_database(settings: DatabaseSettings) -> None:
    _validate_postgres_settings(settings)
    admin_manager: DatabaseManager | None = None
    try:
        admin_settings = clone_settings_with_database(settings, "postgres")
        admin_manager = DatabaseManager(admin_settings)
        admin_engine = admin_manager.engine.execution_options(
            isolation_level="AUTOCOMMIT"
        )
        with admin_engine.connect() as connection:
            exists = connection.execute(
                build_postgres_database_exists_sql(),
                {"name": settings.database_name},
            ).scalar()
            if not exists:
                connection.execute(
                    build_postgres_create_database_sql(settings.database_name)
                )

        _create_schema(settings, f"PostgreSQL database {settings.database_name}")
    finally:
        if admin_manager is not None:
            admin_manager.dispose()

###############################################################################
def prepare_database_for_startup(settings: DatabaseSettings | None = None) -> None:
    database_settings = settings or get_server_settings().database
    if database_settings.embedded_database:
        initialize_sqlite_database(database_settings)
    else:
        verify_postgres_database(database_settings)

###############################################################################
def run_database_initialization(settings: DatabaseSettings | None = None) -> None:
    database_settings = settings or get_server_settings().database
    if database_settings.embedded_database:
        initialize_sqlite_database(database_settings)
    else:
        initialize_postgres_database(database_settings)

###############################################################################
def initialize_database(settings: DatabaseSettings | None = None) -> None:
    try:
        run_database_initialization(settings)
    except (SQLAlchemyError, RuntimeError, ValueError) as exc:
        logger.error("Database initialization failed (%s).", type(exc).__name__)
        raise RuntimeError(
            "Database initialization failed; verify the configured database is reachable."
        ) from None


__all__ = [
    "initialize_database",
    "initialize_postgres_database",
    "initialize_sqlite_database",
    "prepare_database_for_startup",
    "run_database_initialization",
    "verify_postgres_database",
]
