# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

from pathlib import Path

import pytest

from server.configurations.bootstrap import load_database_config, load_storage_root


BOOTSTRAP_ENV_KEYS = (
    "ADSMOD_STORAGE_ROOT",
    "EMBEDDED_DATABASE",
    "DATABASE_ENGINE",
    "DATABASE_HOST",
    "DATABASE_PORT",
    "DATABASE_NAME",
    "DATABASE_USERNAME",
    "DATABASE_PASSWORD",
    "DATABASE_SSL",
    "DATABASE_SSL_CA",
    "DATABASE_CONNECT_TIMEOUT",
    "DATABASE_SQLITE_PATH",
)


def _clear_bootstrap_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in BOOTSTRAP_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_bootstrap_loads_database_connection_values_from_dotenv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_bootstrap_environment(monkeypatch)
    env_path = tmp_path / ".env"
    env_path.write_text(
        "EMBEDDED_DATABASE=false\n"
        "DATABASE_ENGINE=postgresql\n"
        "DATABASE_HOST=db.example.test\n"
        "DATABASE_PORT=6543\n"
        "DATABASE_NAME=adsmod\n"
        "DATABASE_USERNAME=adsmod_user\n"
        "DATABASE_PASSWORD=secret-marker\n"
        "DATABASE_SSL=true\n"
        "DATABASE_CONNECT_TIMEOUT=15\n"
        "DATABASE_SQLITE_PATH=local/settings.db\n"
        "runtime.backend_port=9000\n",
        encoding="utf-8",
    )

    database = load_database_config(dotenv_path=env_path)

    assert database.embedded_database is False
    assert database.engine == "postgresql"
    assert database.host == "db.example.test"
    assert database.port == 6543
    assert database.database_name == "adsmod"
    assert database.username == "adsmod_user"
    assert database.password == "secret-marker"
    assert database.ssl is True
    assert database.connect_timeout == 15
    assert database.sqlite_path == "local/settings.db"


def test_invalid_bootstrap_values_fail_without_disclosing_password(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_bootstrap_environment(monkeypatch)
    env_path = tmp_path / ".env"
    env_path.write_text(
        "EMBEDDED_DATABASE=perhaps\nDATABASE_PASSWORD=private-marker\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="settings/.env") as error:
        load_database_config(dotenv_path=env_path)
    assert "private-marker" not in str(error.value)


def test_storage_root_expands_windows_local_app_data_template(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_bootstrap_environment(monkeypatch)
    local_data = tmp_path / "Local App Data"
    monkeypatch.setenv("LOCALAPPDATA", str(local_data))

    assert load_storage_root(dotenv_path=tmp_path / "missing.env") == (
        local_data / "ADSMOD"
    ).resolve()
