# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from server.configurations.bootstrap import (
    load_database_config,
    load_storage_root,
)
from server.configurations.database_config import database_config_path, write_database_config
from server.configurations.settings import DatabaseConfig


DATABASE_ENV_KEYS = (
    "EMBEDDED_DATABASE", "DATABASE_ENGINE", "DATABASE_HOST", "DATABASE_PORT",
    "DATABASE_NAME", "DATABASE_USERNAME", "DATABASE_PASSWORD", "DATABASE_SSL",
    "DATABASE_SSL_CA", "DATABASE_CONNECT_TIMEOUT", "DATABASE_SQLITE_PATH",
)


###############################################################################
def _clear_database_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (*DATABASE_ENV_KEYS, "ADSMOD_STORAGE_ROOT"):
        monkeypatch.delenv(key, raising=False)


###############################################################################
def test_database_config_migrates_dotenv_once_and_protects_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_database_environment(monkeypatch)
    env_path = tmp_path / ".env"
    root = tmp_path / "application-data"
    env_path.write_text(
        "ADSMOD_STORAGE_ROOT=unused\n"
        "EMBEDDED_DATABASE=false\n"
        "DATABASE_ENGINE=postgresql\n"
        "DATABASE_HOST=db.example.test\n"
        "DATABASE_PORT=6543\n"
        "DATABASE_NAME=adsmod\n"
        "DATABASE_USERNAME=adsmod_user\n"
        "DATABASE_PASSWORD=private-marker\n"
        "DATABASE_SSL=true\n"
        "DATABASE_CONNECT_TIMEOUT=15\n"
        "DATABASE_SQLITE_PATH=local/settings.db\n"
        "unrelated.setting=kept\n",
        encoding="utf-8",
    )

    database = load_database_config(dotenv_path=env_path, storage_root=root)
    persisted = database_config_path(root).read_text(encoding="utf-8")
    restarted = load_database_config(dotenv_path=env_path, storage_root=root)

    assert database.embedded_database is False
    assert database.engine == "postgresql"
    assert database.host == "db.example.test"
    assert database.port == 6543
    assert database.database_name == "adsmod"
    assert database.username == "adsmod_user"
    assert database.password == "private-marker"
    assert database.ssl is True
    assert database.connect_timeout == 15
    assert database.sqlite_path == "local/settings.db"
    assert restarted == database
    assert "private-marker" not in persisted
    assert "password_protected" in persisted
    env_text = env_path.read_text(encoding="utf-8")
    assert "DATABASE_" not in env_text
    assert "EMBEDDED_DATABASE" not in env_text
    assert "unrelated.setting=kept" in env_text


###############################################################################
def test_database_config_file_is_authoritative_over_stale_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_database_environment(monkeypatch)
    root = tmp_path / "application-data"
    env_path = tmp_path / ".env"
    write_database_config(DatabaseConfig(), root)
    env_path.write_text("DATABASE_PASSWORD=stale-secret\n", encoding="utf-8")
    monkeypatch.setenv("DATABASE_HOST", "stale-host")

    database = load_database_config(dotenv_path=env_path, storage_root=root)

    assert database.embedded_database is True
    assert database.host is None
    assert database.password is None
    assert "stale-secret" not in database_config_path(root).read_text(encoding="utf-8")


###############################################################################
def test_invalid_legacy_database_values_fail_without_disclosing_password(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_database_environment(monkeypatch)
    env_path = tmp_path / ".env"
    root = tmp_path / "application-data"
    env_path.write_text(
        "EMBEDDED_DATABASE=perhaps\nDATABASE_PASSWORD=private-marker\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as error:
        load_database_config(dotenv_path=env_path, storage_root=root)
    assert "private-marker" not in str(error.value)
    assert not database_config_path(root).exists()


###############################################################################
def test_invalid_database_file_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "application-data"
    root.mkdir()
    path = database_config_path(root)
    path.write_text(json.dumps({"schema_version": 99}), encoding="utf-8")
    if os.name != "nt":
        path.chmod(0o600)

    with pytest.raises(ValueError, match="database.json"):
        load_database_config(dotenv_path=tmp_path / "missing.env", storage_root=root)


###############################################################################
def test_storage_root_expands_windows_local_app_data_template(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_database_environment(monkeypatch)
    local_data = tmp_path / "Local App Data"
    monkeypatch.setenv("LOCALAPPDATA", str(local_data))

    assert load_storage_root(dotenv_path=tmp_path / "missing.env") == (
        local_data / "ADSMOD"
    ).resolve()
