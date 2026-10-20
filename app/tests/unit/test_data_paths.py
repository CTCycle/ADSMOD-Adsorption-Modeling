# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from pathlib import Path

from server.configurations.settings import AdsmodConfig, DatabaseConfig, UserSettings
from server.repositories.database.manager import resolve_sqlite_path
from server.common.path import resolve_database_path

###############################################################################
def test_relative_database_path_follows_the_canonical_storage_root(
    tmp_path: Path,
) -> None:
    database = DatabaseConfig(embedded_database=True, sqlite_path="data/database.db")

    assert (
        resolve_sqlite_path(database, storage_root=tmp_path)
        == (tmp_path / "data" / "database.db").resolve()
    )

    config = AdsmodConfig.assemble(
        database=database,
        storage_root=tmp_path,
        settings=UserSettings(),
    )
    assert (
        resolve_database_path(config) == (tmp_path / "data" / "database.db").resolve()
    )
