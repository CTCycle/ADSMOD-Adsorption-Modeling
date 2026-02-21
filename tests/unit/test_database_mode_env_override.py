from __future__ import annotations

from ADSMOD.server.configurations.server import build_database_settings


# -------------------------------------------------------------------------
def test_db_embedded_env_override_enabled(monkeypatch):
    monkeypatch.setenv("DB_EMBEDDED", "true")
    settings = build_database_settings({})

    assert settings.embedded_database is True
    assert settings.engine is None
    assert settings.host is None
    assert settings.port is None
    assert settings.database_name is None


# -------------------------------------------------------------------------
def test_db_embedded_env_override_disabled_uses_external_values(monkeypatch):
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

    settings = build_database_settings({})

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


# -------------------------------------------------------------------------
def test_db_settings_do_not_read_json_payload(monkeypatch):
    payload = {
        "embedded_database": False,
        "engine": "postgres",
        "host": "payload-host.example",
        "port": 7777,
        "database_name": "payload_db",
        "username": "payload_user",
        "password": "payload_password",
        "ssl": True,
        "ssl_ca": "/tmp/payload-ca.pem",
        "connect_timeout": 99,
        "insert_batch_size": 1234,
    }

    monkeypatch.setenv("DB_EMBEDDED", "false")
    monkeypatch.delenv("DB_ENGINE", raising=False)
    monkeypatch.delenv("DB_HOST", raising=False)
    monkeypatch.delenv("DB_PORT", raising=False)
    monkeypatch.delenv("DB_NAME", raising=False)
    monkeypatch.delenv("DB_USER", raising=False)
    monkeypatch.delenv("DB_PASSWORD", raising=False)
    monkeypatch.delenv("DB_SSL", raising=False)
    monkeypatch.delenv("DB_SSL_CA", raising=False)
    monkeypatch.delenv("DB_CONNECT_TIMEOUT", raising=False)
    monkeypatch.delenv("DB_INSERT_BATCH_SIZE", raising=False)

    settings = build_database_settings(payload)

    assert settings.engine == "postgres"
    assert settings.host == "localhost"
    assert settings.port == 5432
    assert settings.database_name == "ADSMOD"
    assert settings.username == "postgres"
    assert settings.password == "admin"
    assert settings.ssl is False
    assert settings.ssl_ca is None
    assert settings.connect_timeout == 30
    assert settings.insert_batch_size == 5000
