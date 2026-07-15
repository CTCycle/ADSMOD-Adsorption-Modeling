from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError

from shared.common.settings import DatabaseSettings, get_server_settings
from shared.common.utils.logger import logger
from shared.repositories.database.manager import DatabaseManager
from shared.repositories.database.sql import build_postgres_create_database_sql, build_postgres_database_exists_sql
from shared.repositories.database.utils import normalize_postgres_engine


def clone_settings_with_database(settings: DatabaseSettings, database_name: str) -> DatabaseSettings:
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


def initialize_sqlite_database(settings: DatabaseSettings) -> None:
    manager = DatabaseManager(settings, create_schema=True)
    try:
        logger.info("Initialized SQLite database schema")
    finally:
        manager.dispose()


def ensure_postgres_database(settings: DatabaseSettings) -> str:
    if not settings.host or not settings.username or not settings.database_name:
        raise ValueError("PostgreSQL host, database name, and username are required.")
    target = settings.database_name
    target_settings = clone_settings_with_database(settings, target)
    try:
        manager = DatabaseManager(target_settings, create_schema=True)
    except SQLAlchemyError:
        admin_url = f"{normalize_postgres_engine(settings.engine)}://{settings.username}:{settings.password or ''}@{settings.host}:{settings.port or 5432}/postgres"
        admin_engine = create_engine(admin_url, isolation_level="AUTOCOMMIT", pool_pre_ping=True, connect_args={"connect_timeout": settings.connect_timeout})
        try:
            with admin_engine.connect() as connection:
                exists = connection.execute(build_postgres_database_exists_sql(), {"name": target}).scalar()
                if not exists:
                    connection.execute(build_postgres_create_database_sql(target))
        finally:
            admin_engine.dispose()
        manager = DatabaseManager(target_settings, create_schema=True)
    finally:
        if "manager" in locals():
            manager.dispose()
    return target


def run_database_initialization() -> None:
    settings = get_server_settings().database
    if settings.embedded_database:
        initialize_sqlite_database(settings)
    else:
        ensure_postgres_database(settings)


def initialize_database() -> None:
    try:
        run_database_initialization()
    except (SQLAlchemyError, ValueError) as exc:
        logger.error("Database initialization failed: %s", exc)
        raise SystemExit(1) from exc


__all__ = ["ensure_postgres_database", "initialize_database", "run_database_initialization"]
