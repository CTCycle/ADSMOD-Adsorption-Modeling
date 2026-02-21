from __future__ import annotations

from ADSMOD.server.configurations.server import build_database_settings


# -------------------------------------------------------------------------
def test_db_embedded_env_override_enabled(monkeypatch):
    payload = {
        "embedded_database": False,
        "engine": "postgres",
        "host": "db.example",
        "port": 5432,
        "database_name": "remote_db",
        "username": "remote_user",
        "password": "remote_password",
        "ssl": True,
        "connect_timeout": 30,
        "insert_batch_size": 5000,
    }

    monkeypatch.setenv("DB_EMBEDDED", "true")
    settings = build_database_settings(payload)

    assert settings.embedded_database is True
    assert settings.engine is None
    assert settings.host is None
    assert settings.port is None
    assert settings.database_name is None


# -------------------------------------------------------------------------
def test_db_embedded_env_override_disabled_uses_external_values(monkeypatch):
    payload = {
        "embedded_database": True,
        "engine": "postgres",
        "host": "localhost",
        "port": 5432,
        "database_name": "ADSMOD",
        "username": "postgres",
        "password": "admin",
        "ssl": False,
        "connect_timeout": 10,
        "insert_batch_size": 1000,
    }

    monkeypatch.setenv("DB_EMBEDDED", "false")
    monkeypatch.setenv("DB_ENGINE", "postgres")
    monkeypatch.setenv("DB_HOST", "cloud-db.example.com")
    monkeypatch.setenv("DB_PORT", "6543")
    monkeypatch.setenv("DB_NAME", "cloud_adsmod")
    monkeypatch.setenv("DB_USER", "cloud_user")
    monkeypatch.setenv("DB_PASSWORD", "cloud_password")
    monkeypatch.setenv("DB_SSL", "true")
    monkeypatch.setenv("DB_CONNECT_TIMEOUT", "45")
    monkeypatch.setenv("DB_INSERT_BATCH_SIZE", "6000")

    settings = build_database_settings(payload)

    assert settings.embedded_database is False
    assert settings.engine == "postgres"
    assert settings.host == "cloud-db.example.com"
    assert settings.port == 6543
    assert settings.database_name == "cloud_adsmod"
    assert settings.username == "cloud_user"
    assert settings.password == "cloud_password"
    assert settings.ssl is True
    assert settings.connect_timeout == 45
    assert settings.insert_batch_size == 6000
