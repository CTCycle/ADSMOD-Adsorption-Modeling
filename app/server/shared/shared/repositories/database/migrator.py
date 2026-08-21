from __future__ import annotations

from dataclasses import dataclass
import time

from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from alembic.runtime.migration import MigrationContext
from sqlalchemy import inspect, text
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.exc import OperationalError, SQLAlchemyError

from shared.common.paths import SERVER_PATH
from shared.common.settings import DatabaseSettings
from shared.common.utils.logger import logger
from shared.repositories.database.legacy_schema import (
    BASELINE_REVISION,
    BASELINE_TABLES,
    missing_baseline_tables,
    schema_matches_baseline,
)
from shared.repositories.database.manager import DatabaseManager


MIGRATION_LOCK_KEY = 8_174_209_531
MIGRATION_CONFIG_PATH = SERVER_PATH / "pyproject.toml"


###############################################################################
class DatabaseMigrationError(RuntimeError):
    """Raised when the database cannot be brought to the packaged head."""


###############################################################################
class LegacySchemaMismatchError(DatabaseMigrationError):
    """Raised when an unversioned database is not the known legacy schema."""


###############################################################################
class MigrationLockTimeoutError(DatabaseMigrationError):
    """Raised when another process holds the migration lock too long."""


###############################################################################
@dataclass(frozen=True)
class MigrationResult:
    backend: str
    before: tuple[str, ...]
    after: tuple[str, ...]
    head: str
    adopted_legacy_schema: bool
    applied_migrations: bool


###############################################################################
def build_alembic_config() -> Config:
    if not MIGRATION_CONFIG_PATH.is_file():
        raise DatabaseMigrationError(
            f"Alembic configuration is missing: {MIGRATION_CONFIG_PATH}"
        )
    config = Config(toml_file=MIGRATION_CONFIG_PATH)
    script = ScriptDirectory.from_config(config)
    heads = tuple(script.get_heads())
    if len(heads) != 1:
        raise DatabaseMigrationError(
            "Alembic migration history must contain exactly one head; "
            f"found {len(heads)}. Merge migration branches before startup."
        )
    if script.get_revision(BASELINE_REVISION) is None:
        raise DatabaseMigrationError(
            f"Alembic baseline revision {BASELINE_REVISION} is missing."
        )
    config.attributes["script_directory"] = script
    config.attributes["head_revision"] = heads[0]
    return config


###############################################################################
def _script_and_head(config: Config) -> tuple[ScriptDirectory, str]:
    script = config.attributes.get("script_directory")
    head = config.attributes.get("head_revision")
    if isinstance(script, ScriptDirectory) and isinstance(head, str):
        return script, head
    script = ScriptDirectory.from_config(config)
    heads = tuple(script.get_heads())
    if len(heads) != 1:
        raise DatabaseMigrationError(
            "Alembic migration history must contain exactly one head; "
            f"found {len(heads)}."
        )
    return script, heads[0]


###############################################################################
def _current_heads(connection: Connection) -> tuple[str, ...]:
    migration_context = MigrationContext.configure(
        connection,
        opts={"version_table": "alembic_version"},
    )
    return tuple(migration_context.get_current_heads())


###############################################################################
def _run_command(config: Config, connection: Connection, action: str, revision: str) -> None:
    config.attributes["connection"] = connection
    if action == "stamp":
        command.stamp(config, revision)
    elif action == "upgrade":
        command.upgrade(config, revision)
    else:  # pragma: no cover - private helper guard
        raise ValueError(f"Unsupported Alembic action: {action}")


###############################################################################
def _validate_known_heads(current: tuple[str, ...], script: ScriptDirectory) -> None:
    unknown = [revision for revision in current if script.get_revision(revision) is None]
    if unknown:
        raise DatabaseMigrationError(
            "Database references revisions that are not packaged: "
            + ", ".join(sorted(unknown))
        )
    if len(current) > 1:
        raise DatabaseMigrationError(
            "Database contains multiple Alembic heads: " + ", ".join(sorted(current))
        )


###############################################################################
def _migrate_locked(
    connection: Connection,
    config: Config,
    settings: DatabaseSettings,
) -> MigrationResult:
    script, head = _script_and_head(config)
    inspector = inspect(connection)
    table_names = set(inspector.get_table_names())
    application_tables = table_names & BASELINE_TABLES
    user_tables = table_names - {"alembic_version"}
    version_table_exists = "alembic_version" in table_names
    before = _current_heads(connection)
    _validate_known_heads(before, script)
    adopted = False

    if not version_table_exists:
        if application_tables:
            missing = sorted(BASELINE_TABLES - application_tables)
            if missing:
                raise LegacySchemaMismatchError(
                    "Unversioned database is missing legacy tables: "
                    + ", ".join(missing)
                )
            if not schema_matches_baseline(inspector):
                raise LegacySchemaMismatchError(
                    "Unversioned database does not exactly match the packaged "
                    "pre-Alembic schema; refusing to stamp or alter it."
                )
            logger.info(
                "Adopting the validated legacy %s schema at Alembic revision %s.",
                settings.engine or "sqlite",
                BASELINE_REVISION,
            )
            _run_command(config, connection, "stamp", BASELINE_REVISION)
            adopted = True
            before = (BASELINE_REVISION,)
        elif user_tables:
            raise LegacySchemaMismatchError(
                "Unversioned database contains tables outside the packaged legacy "
                "schema; refusing to stamp or alter it."
            )
        else:
            logger.info("Database has no application tables; applying the Alembic baseline.")
    elif not before and application_tables:
        raise DatabaseMigrationError(
            "Database has an empty alembic_version table alongside application tables; "
            "it may contain an interrupted migration. Restore or repair it manually."
        )

    if before == (head,):
        missing = missing_baseline_tables(inspector)
        if missing:
            raise DatabaseMigrationError(
                "Database is stamped at Alembic head but is missing tables: "
                + ", ".join(missing)
            )
        logger.info("Database is already at Alembic head %s; no migrations needed.", head)
        return MigrationResult(
            backend="sqlite" if settings.embedded_database else "postgres",
            before=before,
            after=before,
            head=head,
            adopted_legacy_schema=adopted,
            applied_migrations=False,
        )

    before_label = ", ".join(before) if before else "base"
    logger.info("Applying Alembic migrations from %s to %s.", before_label, head)
    _run_command(config, connection, "upgrade", head)
    after = _current_heads(connection)
    _validate_known_heads(after, script)
    if after != (head,):
        raise DatabaseMigrationError(
            f"Alembic upgrade completed with current revisions {after!r}; expected {head!r}."
        )
    logger.info("Alembic migrations synchronized database at revision %s.", head)
    return MigrationResult(
        backend="sqlite" if settings.embedded_database else "postgres",
        before=before,
        after=after,
        head=head,
        adopted_legacy_schema=adopted,
        applied_migrations=True,
    )


###############################################################################
def _acquire_postgres_lock(connection: Connection, timeout_seconds: int) -> None:
    deadline = time.monotonic() + max(1, timeout_seconds)
    while True:
        locked = connection.execute(
            text("SELECT pg_try_advisory_xact_lock(:lock_key)"),
            {"lock_key": MIGRATION_LOCK_KEY},
        ).scalar()
        if locked:
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise MigrationLockTimeoutError(
                f"Timed out after {timeout_seconds}s waiting for the PostgreSQL migration lock."
            )
        time.sleep(min(0.1, remaining))


###############################################################################
def migrate_engine(
    engine: Engine,
    settings: DatabaseSettings,
    *,
    config: Config | None = None,
) -> MigrationResult:
    alembic_config = config or build_alembic_config()
    backend = "sqlite" if settings.embedded_database else "postgres"
    started = time.perf_counter()
    logger.info("Checking %s database schema and Alembic revision.", backend)

    if backend == "sqlite":
        with engine.connect() as connection:
            # Python's modern sqlite3 autocommit=False mode begins a DBAPI
            # transaction before the first statement. Temporarily switch the
            # driver to autocommit so BEGIN IMMEDIATE can acquire the writer
            # lock explicitly, then restore normal application semantics.
            driver_connection = connection.connection.driver_connection
            driver_connection.autocommit = True
            try:
                connection.exec_driver_sql("BEGIN IMMEDIATE")
            except OperationalError as exc:
                connection.rollback()
                driver_connection.autocommit = False
                driver_connection.rollback()
                raise MigrationLockTimeoutError(
                    "Timed out waiting for the SQLite migration write lock."
                ) from exc
            try:
                result = _migrate_locked(
                    connection,
                    alembic_config,
                    settings,
                )
            except Exception:
                connection.exec_driver_sql("ROLLBACK")
                connection.rollback()
                driver_connection.autocommit = False
                driver_connection.rollback()
                raise
            else:
                try:
                    connection.exec_driver_sql("COMMIT")
                    connection.commit()
                finally:
                    driver_connection.autocommit = False
                    driver_connection.rollback()
    else:
        with engine.connect() as connection:
            transaction = connection.begin()
            try:
                _acquire_postgres_lock(connection, settings.connect_timeout)
                result = _migrate_locked(connection, alembic_config, settings)
            except Exception:
                transaction.rollback()
                raise
            else:
                transaction.commit()

    elapsed = time.perf_counter() - started
    logger.info(
        "Database migration status=success backend=%s before=%s after=%s head=%s "
        "adopted_legacy=%s applied_migrations=%s lock_timeout=%ss elapsed=%.2fs.",
        backend,
        ",".join(result.before) or "base",
        ",".join(result.after) or "base",
        result.head,
        result.adopted_legacy_schema,
        result.applied_migrations,
        settings.connect_timeout,
        elapsed,
    )
    return result


###############################################################################
def migrate_database(settings: DatabaseSettings) -> MigrationResult:
    manager: DatabaseManager | None = None
    try:
        manager = DatabaseManager(settings)
        return migrate_engine(manager.engine, settings)
    except DatabaseMigrationError:
        raise
    except (SQLAlchemyError, OSError, ValueError) as exc:
        raise DatabaseMigrationError(
            "Database migration failed; verify database connectivity and schema state."
        ) from exc
    finally:
        if manager is not None:
            manager.dispose()


__all__ = [
    "DatabaseMigrationError",
    "LegacySchemaMismatchError",
    "MigrationLockTimeoutError",
    "MigrationResult",
    "build_alembic_config",
    "migrate_database",
    "migrate_engine",
]
