from pathlib import Path

from shared.common.paths import DEFAULT_RESOURCES_DIR, ROOT_PATH, resolve_resources_dir
from shared.common.settings import DatabaseSettings
from shared.repositories.database import manager


###############################################################################
def test_resources_dir_defaults_to_app_resources() -> None:
    assert resolve_resources_dir("") == DEFAULT_RESOURCES_DIR.resolve()


###############################################################################
def test_relative_resources_dir_is_resolved_from_repository_root(tmp_path: Path) -> None:
    configured_path = Path("assets") / "QA" / tmp_path.name

    assert resolve_resources_dir(configured_path) == (ROOT_PATH / configured_path).resolve()


###############################################################################
def test_canonical_sqlite_path_follows_resource_dir_override(
    tmp_path: Path,
    monkeypatch,
) -> None:
    custom_resources_dir = tmp_path / "resources"
    monkeypatch.setattr(manager, "RESOURCES_DIR", custom_resources_dir)
    settings = DatabaseSettings(
        embedded_database=True,
        engine=None,
        host=None,
        port=None,
        database_name=None,
        username=None,
        password=None,
        ssl=False,
        ssl_ca=None,
        connect_timeout=30,
        insert_batch_size=100,
        sqlite_path="app/resources/database.db",
    )

    assert manager.resolve_sqlite_path(settings) == custom_resources_dir / "database.db"
