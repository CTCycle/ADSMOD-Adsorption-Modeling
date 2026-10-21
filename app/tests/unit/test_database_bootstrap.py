# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

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
