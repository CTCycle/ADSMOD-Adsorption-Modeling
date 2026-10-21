# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import sqlite3

from alembic import command
import pytest
from sqlalchemy import inspect

from server.configurations.settings import DatabaseConfig
from server.repositories.database.manager import DatabaseManager
from server.repositories.database.migrator import (
    DatabaseMigrationError,
    INITIAL_REVISION,
    build_alembic_config,
    migrate_database,
    migrate_engine,
)
from server.repositories.schemas.models import Base


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

    with pytest.raises(DatabaseMigrationError, match="Recreate the database"):
        migrate_database(_sqlite_settings(str(path)))

    with sqlite3.connect(path) as connection:
        assert connection.execute(
            "SELECT version_num FROM alembic_version"
        ).fetchone() == ("23f1110c64a9",)
        assert connection.execute(
            "SELECT value FROM retained_user_data WHERE id = 1"
        ).fetchone() == ("preserve",)
