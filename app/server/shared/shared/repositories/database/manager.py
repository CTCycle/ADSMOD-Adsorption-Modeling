from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from shared.common.constants import DATABASE_FILENAME
from shared.common.paths import RESOURCES_DIR
from shared.common.settings import DatabaseSettings
from shared.common.utils.logger import logger
from shared.repositories.schemas.models import Base
from shared.repositories.database.utils import normalize_postgres_engine


class DatabaseManager:
    """Own the single engine, session factory, and transaction boundary."""

    def __init__(self, settings: DatabaseSettings, *, create_schema: bool = False) -> None:
        self.settings = settings
        self.backend = "sqlite" if settings.embedded_database else self._normalize_backend(settings.engine)
        self.engine = self._create_engine()
        self.session_factory = sessionmaker(bind=self.engine, future=True, expire_on_commit=False)
        if create_schema:
            Base.metadata.create_all(self.engine)

    @staticmethod
    def _normalize_backend(engine: str | None) -> str:
        value = (engine or "postgres").lower()
        if value in {"postgres", "postgresql", "postgresql+psycopg", "postgresql+psycopg2"}:
            return "postgres"
        raise ValueError(f"Unsupported database engine: {engine}")

    def _create_engine(self) -> Engine:
        if self.backend == "sqlite":
            if self.settings.sqlite_path == ":memory:":
                engine = create_engine("sqlite:///:memory:", future=True, connect_args={"check_same_thread": False})
                event.listen(engine, "connect", self._configure_sqlite)
                return engine
            path = Path(self.settings.sqlite_path) if self.settings.sqlite_path else RESOURCES_DIR / DATABASE_FILENAME
            path.parent.mkdir(parents=True, exist_ok=True)
            engine = create_engine(f"sqlite:///{path}", future=True, connect_args={"timeout": self.settings.connect_timeout, "check_same_thread": False})
            event.listen(engine, "connect", self._configure_sqlite)
            return engine
        if not self.settings.host or not self.settings.database_name or not self.settings.username:
            raise ValueError("PostgreSQL host, database name, and username are required.")
        engine_name = normalize_postgres_engine(self.settings.engine)
        url = f"{engine_name}://{self.settings.username}:{self.settings.password or ''}@{self.settings.host}:{self.settings.port or 5432}/{self.settings.database_name}"
        connect_args: dict[str, Any] = {"connect_timeout": self.settings.connect_timeout, "client_encoding": "utf8"}
        if self.settings.ssl:
            connect_args["sslmode"] = "require"
            if self.settings.ssl_ca:
                connect_args["sslrootcert"] = self.settings.ssl_ca
        return create_engine(url, future=True, connect_args=connect_args, pool_pre_ping=True)

    @staticmethod
    def _configure_sqlite(dbapi_connection: Any, connection_record: Any) -> None:
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA busy_timeout=30000")
            if dbapi_connection.execute("PRAGMA database_list").fetchone()[2] != ":memory:":
                cursor.execute("PRAGMA journal_mode=WAL")
        finally:
            cursor.close()

    @contextmanager
    def transaction(self) -> Iterator[Session]:
        with self.session_factory() as session:
            try:
                yield session
                session.commit()
            except Exception:
                session.rollback()
                raise

    def session(self) -> Session:
        return self.session_factory()

    def dispose(self) -> None:
        logger.debug("Disposing %s database engine", self.backend)
        self.engine.dispose()


__all__ = ["DatabaseManager"]
