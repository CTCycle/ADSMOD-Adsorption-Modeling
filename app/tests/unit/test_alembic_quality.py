# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import sqlite3

from alembic import command
from sqlalchemy import inspect

from server.configurations.settings import DatabaseConfig
from server.repositories.database.manager import DatabaseManager
from server.repositories.database.migrator import (
    build_alembic_config,
    migrate_engine,
)
from server.repositories.schemas.models import Base

###############################################################################
def _sqlite_settings(path: str) -> DatabaseConfig:
    return DatabaseConfig(
        embedded_database=True,
        connect_timeout=5,
        insert_batch_size=100,
        sqlite_path=path,
    )

###############################################################################
def test_packaged_history_has_one_head_and_no_pending_operations(tmp_path) -> None:  # type: ignore[no-untyped-def]
    settings = _sqlite_settings(str(tmp_path / "quality.db"))
    manager = DatabaseManager(settings)
    try:
        result = migrate_engine(manager.engine, settings)
        config = build_alembic_config()
        with manager.engine.connect() as connection:
            config.attributes["connection"] = connection
            command.check(config)
            assert set(inspect(connection).get_table_names()) >= set(
                Base.metadata.tables
            )
        assert result.after == (result.head,)
        assert config.attributes["script_directory"].get_heads() == [result.head]
    finally:
        manager.dispose()


def test_public_data_migration_preserves_existing_source_records(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "populated-before-public-sync.db"
    settings = _sqlite_settings(str(path))
    config = build_alembic_config()
    config.attributes["database"] = settings
    command.upgrade(config, "20260924_fitting_cancel")

    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            INSERT INTO data_sources (
                key, name, description, homepage_url, capabilities, enabled,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "legacy-test-source",
                "Legacy test source",
                "Existing source",
                "https://adsorption.nist.gov/",
                '["adsorption"]',
                1,
                "2026-01-01T00:00:00+00:00",
                "2026-01-01T00:00:00+00:00",
            ),
        )
        source_id = connection.execute(
            "SELECT id FROM data_sources WHERE key = 'legacy-test-source'"
        ).fetchone()[0]
        connection.execute(
            """
            INSERT INTO source_records (
                source_id, record_type, external_id, transform_version,
                retrieved_at, raw_metadata
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                source_id,
                "adsorption",
                "existing-experiment-1",
                "public-data-v1",
                "2026-01-01T00:00:00+00:00",
                "{}",
            ),
        )

    manager = DatabaseManager(settings)
    try:
        result = migrate_engine(manager.engine, settings)
    finally:
        manager.dispose()

    with sqlite3.connect(path) as connection:
        assert connection.execute(
            "SELECT health_status FROM data_sources WHERE key = 'legacy-test-source'"
        ).fetchone()[0] == "unknown"
        assert connection.execute(
            "SELECT external_id FROM source_records"
        ).fetchall() == [("existing-experiment-1",)]
        assert connection.execute(
            "SELECT last_successful_sync_at FROM data_sources WHERE key = 'legacy-test-source'"
        ).fetchone()[0] is None
    assert result.after == (result.head,)
