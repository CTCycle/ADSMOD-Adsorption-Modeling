# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations
import json
import os
import pytest
import sqlite3
from pathlib import Path
from server.configurations.bootstrap import (
    RETIRED_DATABASE_KEYS,
    load_database_configuration,
    load_runtime_settings,
    load_storage_root,
    persist_database_configuration,
)
from server.configurations.database_config import (
    database_config_path,
    write_database_config,
)
from server.configurations.settings import DatabaseConfig
from concurrent.futures import ThreadPoolExecutor
from sqlalchemy.exc import SQLAlchemyError
from server.repositories.database import initializer, migrator
from server.repositories.database.initializer import (
    DatabaseMigrationError,
    MigrationLockTimeoutError,
)
from server.repositories.database.manager import DatabaseManager
from server.repositories.schemas.models import Base
from alembic import command
from sqlalchemy import inspect
from server.repositories.database.migrator import (
    DatabaseMigrationError as MigratorDatabaseMigrationError,
    INITIAL_REVISION,
    build_alembic_config,
    migrate_database,
    migrate_engine,
)


# Database bootstrap

RUNTIME_ENV_KEYS = (
    "ADSMOD_HOST",
    "ADSMOD_BACKEND_PORT",
    "ADSMOD_FRONTEND_PORT",
    "ADSMOD_CORS_ORIGINS",
    "ADSMOD_CORS_ALLOW_CREDENTIALS",
    "ADSMOD_CORS_ALLOW_METHODS",
    "ADSMOD_CORS_ALLOW_HEADERS",
)


###############################################################################
def _clear_database_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (*RETIRED_DATABASE_KEYS, *RUNTIME_ENV_KEYS, "ADSMOD_STORAGE_ROOT"):
        monkeypatch.delenv(key, raising=False)


###############################################################################
def test_missing_config_uses_current_defaults_and_scrubs_retired_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_database_environment(monkeypatch)
    env_path = tmp_path / ".env"
    root = tmp_path / "application-data"
    env_path.write_text(
        "DATABASE_HOST=retired-host\n"
        "DATABASE_PASSWORD=private-marker\n"
        "EMBEDDED_DATABASE=false\n"
        "unrelated.setting=kept\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DATABASE_PASSWORD", "process-secret")

    loaded = load_database_configuration(dotenv_path=env_path, storage_root=root)

    assert loaded.config == DatabaseConfig()
    assert loaded.needs_persist is True
    assert not database_config_path(root).exists()
    env_text = env_path.read_text(encoding="utf-8")
    assert "DATABASE_" not in env_text
    assert "EMBEDDED_DATABASE" not in env_text
    assert "unrelated.setting=kept" in env_text
    assert all(key not in os.environ for key in RETIRED_DATABASE_KEYS)


###############################################################################
def test_database_config_file_is_authoritative_over_retired_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_database_environment(monkeypatch)
    root = tmp_path / "application-data"
    env_path = tmp_path / ".env"
    expected = DatabaseConfig()
    write_database_config(expected, root)
    env_path.write_text(
        "DATABASE_PASSWORD=stale-secret\nDATABASE_HOST=stale-host\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DATABASE_HOST", "process-stale-host")

    loaded = load_database_configuration(dotenv_path=env_path, storage_root=root)

    assert loaded.config == expected
    assert loaded.needs_persist is False
    assert "DATABASE_" not in env_path.read_text(encoding="utf-8")
    assert all(key not in os.environ for key in RETIRED_DATABASE_KEYS)
    assert "stale-secret" not in database_config_path(root).read_text(encoding="utf-8")


###############################################################################
def test_persisted_config_round_trips_and_protects_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_database_environment(monkeypatch)
    env_path = tmp_path / ".env"
    env_path.write_text("DATABASE_PASSWORD=private-marker\n", encoding="utf-8")
    config = DatabaseConfig(
        embedded_database=False,
        engine="postgresql",
        host="db.example.test",
        port=6543,
        database_name="adsmod",
        username="adsmod_user",
        password="private-marker",
        ssl=True,
        connect_timeout=15,
    )

    path = persist_database_configuration(
        config, tmp_path / "application-data", dotenv_path=env_path
    )
    persisted = path.read_text(encoding="utf-8")
    loaded = load_database_configuration(dotenv_path=env_path, storage_root=path.parent)

    assert loaded.config == config
    assert loaded.needs_persist is False
    assert "private-marker" not in persisted
    assert "password_protected" in persisted
    assert "DATABASE_" not in env_path.read_text(encoding="utf-8")


###############################################################################
def test_invalid_database_file_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "application-data"
    root.mkdir()
    path = database_config_path(root)
    path.write_text(json.dumps({"schema_version": 99}), encoding="utf-8")
    if os.name != "nt":
        path.chmod(0o600)

    with pytest.raises(ValueError, match="database.json"):
        load_database_configuration(
            dotenv_path=tmp_path / "missing.env", storage_root=root
        )


###############################################################################
def test_storage_root_expands_windows_local_app_data_template(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_database_environment(monkeypatch)
    local_data = tmp_path / "Local App Data"
    monkeypatch.setenv("LOCALAPPDATA", str(local_data))

    assert (
        load_storage_root(dotenv_path=tmp_path / "missing.env")
        == (local_data / "ADSMOD").resolve()
    )


###############################################################################
def test_runtime_environment_defaults_and_derived_endpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_database_environment(monkeypatch)

    runtime = load_runtime_settings(dotenv_path=tmp_path / "missing.env")

    assert runtime.host == "127.0.0.1"
    assert runtime.backend_port == 6045
    assert runtime.frontend_port == 5173
    assert runtime.backend_url == "http://127.0.0.1:6045"
    assert runtime.frontend_url == "http://127.0.0.1:5173"
    assert runtime.cors_origins == (
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    )
    assert runtime.cors_allow_credentials is True
    assert runtime.cors_allow_methods == ("*",)
    assert runtime.cors_allow_headers == ("*",)


###############################################################################
def test_runtime_dotenv_parsing_and_process_environment_precedence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_database_environment(monkeypatch)
    env_path = tmp_path / ".env"
    env_path.write_text(
        "ADSMOD_HOST=localhost\n"
        "ADSMOD_BACKEND_PORT=6046\n"
        "ADSMOD_FRONTEND_PORT=5174\n"
        "ADSMOD_CORS_ORIGINS=https://localhost:5174,https://example.test\n"
        "ADSMOD_CORS_ALLOW_CREDENTIALS=false\n"
        "ADSMOD_CORS_ALLOW_METHODS=GET, PUT\n"
        "ADSMOD_CORS_ALLOW_HEADERS=X-Trace, Content-Type\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ADSMOD_BACKEND_PORT", "6047")

    runtime = load_runtime_settings(dotenv_path=env_path)

    assert runtime.host == "localhost"
    assert runtime.backend_port == 6047
    assert runtime.frontend_port == 5174
    assert runtime.backend_url == "http://localhost:6047"
    assert runtime.cors_origins == ("https://localhost:5174", "https://example.test")
    assert runtime.cors_allow_credentials is False
    assert runtime.cors_allow_methods == ("GET", "PUT")
    assert runtime.cors_allow_headers == ("X-Trace", "Content-Type")


###############################################################################
@pytest.mark.parametrize(
    "environment",
    [
        {"ADSMOD_HOST": "0.0.0.0"},
        {"ADSMOD_BACKEND_PORT": "1023"},
        {"ADSMOD_FRONTEND_PORT": "65536"},
        {"ADSMOD_BACKEND_PORT": "5173"},
        {"ADSMOD_CORS_ORIGINS": "ftp://example.test"},
        {"ADSMOD_CORS_ORIGINS": "http://example.test:65536"},
        {"ADSMOD_CORS_ALLOW_CREDENTIALS": "sometimes"},
    ],
)
def test_invalid_runtime_environment_values_fail_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    environment: dict[str, str],
) -> None:
    _clear_database_environment(monkeypatch)
    for key, value in environment.items():
        monkeypatch.setenv(key, value)

    with pytest.raises(ValueError):
        load_runtime_settings(dotenv_path=tmp_path / "missing.env")


# Database initialization


###############################################################################
def sqlite_config(path: Path, *, timeout: int = 5) -> DatabaseConfig:
    return DatabaseConfig(
        embedded_database=True,
        connect_timeout=timeout,
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
    database = sqlite_config(path)

    first = initializer.prepare_database_for_startup(database)
    second = initializer.initialize_database(database)

    assert first.applied_migrations is True
    assert second.applied_migrations is False
    assert "alembic_version" in table_names(path)
    assert len(table_names(path) & set(Base.metadata.tables)) == len(
        Base.metadata.tables
    )


###############################################################################
def test_sqlite_empty_existing_file_is_initialized(tmp_path: Path) -> None:
    path = tmp_path / "empty.db"
    path.touch()

    result = migrator.migrate_database(sqlite_config(path))

    assert result.after == (result.head,)
    assert table_names(path) >= {"alembic_version", "datasets"}


###############################################################################
def test_nonempty_unversioned_database_is_rejected_without_inference(
    tmp_path: Path,
) -> None:
    path = tmp_path / "unversioned.db"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE unrelated (id INTEGER PRIMARY KEY)")

    with pytest.raises(DatabaseMigrationError, match="Non-empty unversioned"):
        migrator.migrate_database(sqlite_config(path))

    assert table_names(path) == {"unrelated"}


###############################################################################
def test_empty_version_table_with_application_tables_fails_safely(
    tmp_path: Path,
) -> None:
    path = tmp_path / "interrupted.db"
    manager = DatabaseManager(sqlite_config(path))
    try:
        Base.metadata.create_all(manager.engine)
    finally:
        manager.dispose()
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)"
        )
        connection.commit()

    with pytest.raises(DatabaseMigrationError, match="empty alembic_version"):
        migrator.migrate_database(sqlite_config(path))


###############################################################################
def test_unknown_alembic_revision_is_rejected_without_schema_inference(
    tmp_path: Path,
) -> None:
    path = tmp_path / "unknown-revision.db"
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)"
        )
        connection.execute(
            "INSERT INTO alembic_version (version_num) VALUES (?)",
            ("unknown_revision",),
        )

    with pytest.raises(DatabaseMigrationError, match="Recreate the database"):
        migrator.migrate_database(sqlite_config(path))

    assert table_names(path) == {"alembic_version"}


###############################################################################
def test_database_stamped_at_head_but_incomplete_is_rejected(
    tmp_path: Path,
) -> None:
    path = tmp_path / "incomplete-head.db"
    head = migrator.build_alembic_config().attributes["head_revision"]
    assert isinstance(head, str)
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)"
        )
        connection.execute(
            "INSERT INTO alembic_version (version_num) VALUES (?)", (head,)
        )

    with pytest.raises(
        DatabaseMigrationError,
        match="stamped at Alembic head but is missing tables",
    ):
        migrator.migrate_database(sqlite_config(path))

    assert table_names(path) == {"alembic_version"}


###############################################################################
def test_failed_migration_rolls_back_schema(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "rollback.db"
    database = sqlite_config(path)
    config = migrator.build_alembic_config()

    def fail_after_ddl(_config, connection, _action, _revision):  # type: ignore[no-untyped-def]
        connection.exec_driver_sql("CREATE TABLE transient_failure (id INTEGER)")
        raise SQLAlchemyError("synthetic migration failure")

    monkeypatch.setattr(migrator, "_run_command", fail_after_ddl)
    manager = DatabaseManager(database)
    try:
        with pytest.raises(SQLAlchemyError, match="synthetic"):
            migrator.migrate_engine(manager.engine, database, config=config)
    finally:
        manager.dispose()

    assert "transient_failure" not in table_names(path)
    assert "alembic_version" not in table_names(path)


###############################################################################
def test_concurrent_sqlite_startup_serializes_migrations(tmp_path: Path) -> None:
    path = tmp_path / "concurrent.db"

    def start_once():  # type: ignore[no-untyped-def]
        return migrator.migrate_database(sqlite_config(path))

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
            migrator.migrate_database(sqlite_config(path, timeout=1))
    finally:
        blocker.rollback()
        blocker.close()


###############################################################################
def test_database_initialization_sanitizes_sqlalchemy_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def raise_database_error(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise SQLAlchemyError("postgresql://postgres:secret@host/database")

    monkeypatch.setattr(
        initializer, "prepare_database_for_startup", raise_database_error
    )

    with pytest.raises(RuntimeError) as error:
        initializer.initialize_database(sqlite_config(tmp_path / "unused.db"))

    assert "secret" not in str(error.value)
    assert "verify" in str(error.value)


###############################################################################
def test_postgres_creation_permission_error_is_actionable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = DatabaseConfig(
        embedded_database=False,
        engine="postgres",
        host="localhost",
        port=5432,
        database_name="adsmod_permission_test",
        username="app",
        password="secret",
        ssl=False,
        connect_timeout=1,
    )

    def fail_to_connect(_database: DatabaseConfig) -> DatabaseManager:
        raise SQLAlchemyError("permission denied to create database")

    monkeypatch.setattr(initializer, "DatabaseManager", fail_to_connect)
    with pytest.raises(DatabaseMigrationError, match="CREATEDB"):
        initializer._ensure_postgres_database_exists(database)


# Alembic quality

RETIRED_TABLES = {"training_datasets", "training_samples"}
RETIRED_SETTINGS_COLUMNS = {
    "runtime_host",
    "runtime_backend_port",
    "runtime_frontend_port",
}


###############################################################################
def _sqlite_settings(path: str) -> DatabaseConfig:
    return DatabaseConfig(
        embedded_database=True,
        connect_timeout=5,
        sqlite_path=path,
    )


###############################################################################
def test_packaged_history_is_one_root_revision_matching_current_models(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    settings = _sqlite_settings(str(tmp_path / "current-schema.db"))
    manager = DatabaseManager(settings)
    try:
        result = migrate_engine(manager.engine, settings)
        config = build_alembic_config()
        script = config.attributes["script_directory"]
        revisions = tuple(script.walk_revisions())

        assert len(revisions) == 1
        assert revisions[0].revision == INITIAL_REVISION == "20261021_current_schema"
        assert revisions[0].down_revision is None
        assert result.after == (INITIAL_REVISION,)
        with manager.engine.connect() as connection:
            table_names = set(inspect(connection).get_table_names())
            application_tables = table_names - {"alembic_version", "sqlite_sequence"}
            assert application_tables == set(Base.metadata.tables)
            assert not RETIRED_TABLES.intersection(application_tables)
            settings_columns = {
                column["name"]
                for column in inspect(connection).get_columns("application_settings")
            }
            assert not RETIRED_SETTINGS_COLUMNS.intersection(settings_columns)
            config.attributes["connection"] = connection
            command.check(config)
    finally:
        manager.dispose()


###############################################################################
def test_unknown_historical_revision_fails_closed_without_dropping_data(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "historical-schema.db"
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL PRIMARY KEY)"
        )
        connection.execute(
            "INSERT INTO alembic_version (version_num) VALUES (?)",
            ("23f1110c64a9",),
        )
        connection.execute(
            "CREATE TABLE retained_user_data (id INTEGER PRIMARY KEY, value TEXT NOT NULL)"
        )
        connection.execute(
            "INSERT INTO retained_user_data (id, value) VALUES (1, 'preserve')"
        )

    with pytest.raises(MigratorDatabaseMigrationError, match="Recreate the database"):
        migrate_database(_sqlite_settings(str(path)))

    with sqlite3.connect(path) as connection:
        assert connection.execute(
            "SELECT version_num FROM alembic_version"
        ).fetchone() == ("23f1110c64a9",)
        assert connection.execute(
            "SELECT value FROM retained_user_data WHERE id = 1"
        ).fetchone() == ("preserve",)
