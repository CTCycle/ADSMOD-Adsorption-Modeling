from __future__ import annotations

from alembic import command
from sqlalchemy import inspect

from shared.common.settings import DatabaseSettings
from shared.repositories.database.manager import DatabaseManager
from shared.repositories.database.migrator import build_alembic_config, migrate_engine
from shared.repositories.database.legacy_schema import (
    BASELINE_TABLES,
    schema_matches_baseline,
)


def _sqlite_settings(path: str) -> DatabaseSettings:
    return DatabaseSettings(
        embedded_database=True,
        engine=None,
        host=None,
        port=None,
        database_name=None,
        username=None,
        password=None,
        ssl=False,
        ssl_ca=None,
        connect_timeout=5,
        insert_batch_size=100,
        sqlite_path=path,
    )


def test_packaged_history_has_one_head_and_no_pending_operations(tmp_path) -> None:  # type: ignore[no-untyped-def]
    settings = _sqlite_settings(str(tmp_path / "quality.db"))
    manager = DatabaseManager(settings)
    try:
        result = migrate_engine(manager.engine, settings)
        config = build_alembic_config()
        with manager.engine.connect() as connection:
            config.attributes["connection"] = connection
            command.check(config)
            assert set(inspect(connection).get_table_names()) >= BASELINE_TABLES
            assert schema_matches_baseline(inspect(connection))
        assert result.after == (result.head,)
        assert config.attributes["script_directory"].get_heads() == [result.head]
    finally:
        manager.dispose()
