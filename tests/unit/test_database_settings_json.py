from __future__ import annotations

import pytest

from ADSMOD.server.configurations.server import build_database_settings


# -------------------------------------------------------------------------
def test_db_embedded_json_configuration() -> None:
    settings = build_database_settings(
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


# -------------------------------------------------------------------------
def test_db_external_json_configuration() -> None:
    settings = build_database_settings(
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


# -------------------------------------------------------------------------
def test_db_settings_use_defaults_when_database_payload_missing() -> None:
    settings = build_database_settings({})

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


# -------------------------------------------------------------------------
def test_db_settings_are_not_env_driven_anymore(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DB_EMBEDDED", "false")
    monkeypatch.setenv("DB_HOST", "env-host.example")

    settings = build_database_settings({"embedded_database": True})

    assert settings.embedded_database is True
    assert settings.host is None


# -------------------------------------------------------------------------
def test_db_settings_reject_insecure_placeholder_password() -> None:
    with pytest.raises(ValueError, match="DB_PASSWORD uses an insecure placeholder"):
        build_database_settings(
            {
                "embedded_database": False,
                "password": "",
            }
        )
