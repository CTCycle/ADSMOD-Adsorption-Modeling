from __future__ import annotations

import pytest

from adsmod_common.config import DatabaseConfig, load_config
from shared.common.paths import CANONICAL_CONFIGURATION_FILE
from shared.common.settings import DatabaseSettings, build_database_settings

###############################################################################
def project(payload: dict[str, object]) -> DatabaseSettings:
    canonical = load_config(CANONICAL_CONFIGURATION_FILE).application.database.model_dump(
        mode="python"
    )
    canonical.update(payload)
    return build_database_settings(DatabaseConfig.model_validate(canonical))

###############################################################################
def test_db_embedded_json_configuration() -> None:
    settings = project(
        {
            "embedded_database": True,
            "connect_timeout": 45,
            "insert_batch_size": 6000,
        }
    )

    assert settings.embedded_database is True
    assert settings.engine is None
    assert settings.host is None
    assert settings.port is None
    assert settings.database_name is None
    assert settings.username is None
    assert settings.password is None
    assert settings.ssl is False
    assert settings.ssl_ca is None
    assert settings.connect_timeout == 45
    assert settings.insert_batch_size == 6000

###############################################################################
def test_db_external_json_configuration() -> None:
    settings = project(
        {
            "embedded_database": False,
            "engine": "postgres",
            "host": "external-db.example.com",
            "port": 6543,
            "database_name": "external_adsmod",
            "username": "external_user",
            "password": "external_password",
            "ssl": True,
            "ssl_ca": "/tmp/ca.pem",
            "connect_timeout": 45,
            "insert_batch_size": 6000,
        }
    )

    assert settings.embedded_database is False
    assert settings.engine == "postgres"
    assert settings.host == "external-db.example.com"
    assert settings.port == 6543
    assert settings.database_name == "external_adsmod"
    assert settings.username == "external_user"
    assert settings.password == "external_password"
    assert settings.ssl is True
    assert settings.ssl_ca == "/tmp/ca.pem"
    assert settings.connect_timeout == 45
    assert settings.insert_batch_size == 6000

###############################################################################
def test_db_settings_use_canonical_values_when_no_override_is_given() -> None:
    settings = project({})

    assert settings.embedded_database is True
    assert settings.engine is None
    assert settings.host is None
    assert settings.port is None
    assert settings.database_name is None
    assert settings.username is None
    assert settings.password is None
    assert settings.ssl is False
    assert settings.ssl_ca is None
    assert settings.connect_timeout == 30
    assert settings.insert_batch_size == 5000

###############################################################################
def test_db_settings_are_not_env_driven_anymore(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DB_EMBEDDED", "false")
    monkeypatch.setenv("DB_HOST", "env-host.example")

    settings = project({"embedded_database": True})

    assert settings.embedded_database is True
    assert settings.host is None

###############################################################################
def test_db_settings_allow_minimal_external_payload() -> None:
    settings = project({"embedded_database": False, "password": ""})
    assert settings.embedded_database is False
    assert settings.password is None
