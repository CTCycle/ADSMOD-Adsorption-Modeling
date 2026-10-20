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


###############################################################################
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


###############################################################################
def test_public_data_navigation_migration_merges_nist_identities(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "populated-before-public-navigation.db"
    settings = _sqlite_settings(str(path))
    config = build_alembic_config()
    config.attributes["database"] = settings
    command.upgrade(config, "20261019_public_data_sync")

    now = "2026-01-01T00:00:00+00:00"
    with sqlite3.connect(path) as connection:
        nist_id = connection.execute(
            "SELECT id FROM data_sources WHERE key = 'nist'"
        ).fetchone()[0]
        connection.execute(
            "INSERT INTO datasets (id, name, normalized_name, source, description, tags, "
            "provenance, created_at, updated_at) "
            "VALUES (1, 'NIST ISODB', 'nist isodb', 'nist', '', '[]', '{}', ?, ?)",
            (now, now),
        )
        connection.executemany(
            "INSERT INTO adsorbates (id, key, name, normalized_name, inchi_key, "
            "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (1, "adsorbate:by-name", "carbon dioxide", "carbon dioxide", None, now, now),
                (
                    2,
                    "adsorbate:by-inchikey",
                    "Carbon Dioxide",
                    "carbon dioxide",
                    "CURLTUGMZLYLDI-UHFFFAOYSA-N",
                    now,
                    now,
                ),
            ],
        )
        connection.execute(
            "INSERT INTO adsorbents (id, key, name, normalized_name, created_at, updated_at) "
            "VALUES (1, 'adsorbent:by-name', 'zeolite 5a', 'zeolite 5a', ?, ?)",
            (now, now),
        )
        connection.execute(
            "INSERT INTO isotherms (id, dataset_id, external_key, name, adsorbent_id, "
            "temperature_original, temperature_original_unit, temperature_k, "
            "pressure_basis, duplicate_policy, conditions, provenance, created_at, "
            "updated_at) VALUES (1, 1, 'iso-1', 'iso-1', 1, 298, 'K', 298, "
            "'absolute', 'reject', '{}', '{}', ?, ?)",
            (now, now),
        )
        connection.execute(
            "INSERT INTO isotherm_components (isotherm_id, position, adsorbate_id) "
            "VALUES (1, 1, 1)"
        )
        connection.executemany(
            "INSERT INTO source_records (id, source_id, record_type, external_id, "
            "transform_version, retrieved_at, raw_metadata) "
            "VALUES (?, ?, ?, ?, 'public-data-v1', ?, '{}')",
            [
                (1, nist_id, "chemical", "CURLTUGMZLYLDI-UHFFFAOYSA-N", now),
                (2, nist_id, "material", "adsorbent:by-name", now),
                (3, nist_id, "adsorption", "iso-1", now),
            ],
        )
        connection.execute(
            "INSERT INTO adsorbate_source_records (source_record_id, adsorbate_id) "
            "VALUES (1, 2)"
        )
        connection.execute(
            "INSERT INTO adsorbent_source_records (source_record_id, adsorbent_id) "
            "VALUES (2, 1)"
        )
        connection.execute(
            "INSERT INTO isotherm_source_records (source_record_id, isotherm_id) "
            "VALUES (3, 1)"
        )
        connection.execute(
            "INSERT INTO public_data_sync_runs (id, source_key, operation, parameters, "
            "checkpoint, created_at, updated_at) "
            "VALUES ('run-1', 'nist', 'nist_category', '{}', '{}', ?, ?)",
            (now, now),
        )
        connection.execute(
            "INSERT INTO public_data_sync_failures (sync_run_id, source_key, reason, "
            "created_at) VALUES ('run-1', 'nist', 'kept', ?)",
            (now,),
        )

    manager = DatabaseManager(settings)
    try:
        result = migrate_engine(manager.engine, settings)
    finally:
        manager.dispose()

    with sqlite3.connect(path) as connection:
        assert connection.execute(
            "SELECT adsorbate_id FROM isotherm_components"
        ).fetchall() == [(2,)]
        assert connection.execute("SELECT id FROM adsorbates").fetchall() == [(2,)]
        assert connection.execute(
            "SELECT record_type, source_url FROM source_records ORDER BY id"
        ).fetchall() == [
            ("chemical", None),
            ("adsorption", "https://adsorption.nist.gov/isodb/api/isotherm/iso-1.json"),
        ]
        assert connection.execute(
            "SELECT count(*) FROM adsorbent_source_records"
        ).fetchone() == (0,)
        assert connection.execute(
            "SELECT reason FROM public_data_sync_failures"
        ).fetchall() == [("kept",)]
    assert result.after == (result.head,)
