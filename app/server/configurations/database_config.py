# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

"""Independent, protected persistence for database connection settings."""

from __future__ import annotations

import base64
import ctypes
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from server.configurations.settings import DatabaseConfig

SCHEMA_VERSION = 1
CONFIG_FILENAME = "database.json"


def database_config_path(storage_root: Path) -> Path:
    return storage_root.resolve() / CONFIG_FILENAME


def _protect_secret(value: str) -> str:
    raw = value.encode("utf-8")
    if os.name == "nt":
        class DataBlob(ctypes.Structure):
            _fields_ = [("cbData", ctypes.c_uint32), ("pbData", ctypes.POINTER(ctypes.c_ubyte))]

        source_buffer = ctypes.create_string_buffer(raw)
        source = DataBlob(len(raw), ctypes.cast(source_buffer, ctypes.POINTER(ctypes.c_ubyte)))
        target = DataBlob()
        if not ctypes.windll.crypt32.CryptProtectData(
            ctypes.byref(source), "ADSMOD database credential", None, None, None, 1, ctypes.byref(target)
        ):
            raise OSError("Windows credential protection failed.")
        try:
            encrypted = ctypes.string_at(target.pbData, target.cbData)
        finally:
            ctypes.windll.kernel32.LocalFree(target.pbData)
        return "dpapi:" + base64.b64encode(encrypted).decode("ascii")
    return "plain:" + base64.b64encode(raw).decode("ascii")


def _unprotect_secret(value: str) -> str:
    prefix, encoded = value.split(":", 1)
    encrypted = base64.b64decode(encoded, validate=True)
    if prefix == "dpapi" and os.name == "nt":
        class DataBlob(ctypes.Structure):
            _fields_ = [("cbData", ctypes.c_uint32), ("pbData", ctypes.POINTER(ctypes.c_ubyte))]

        source_buffer = ctypes.create_string_buffer(encrypted)
        source = DataBlob(len(encrypted), ctypes.cast(source_buffer, ctypes.POINTER(ctypes.c_ubyte)))
        target = DataBlob()
        if not ctypes.windll.crypt32.CryptUnprotectData(
            ctypes.byref(source), None, None, None, None, 1, ctypes.byref(target)
        ):
            raise ValueError("Database credential could not be decrypted for this Windows user.")
        try:
            raw = ctypes.string_at(target.pbData, target.cbData)
        finally:
            ctypes.windll.kernel32.LocalFree(target.pbData)
    elif prefix == "plain" and os.name != "nt":
        raw = encrypted
    else:
        raise ValueError("Database credential protection does not match this operating system.")
    return raw.decode("utf-8")


def _payload(config: DatabaseConfig) -> dict[str, Any]:
    if config.embedded_database and config.sqlite_path == ":memory:":
        raise ValueError("In-memory SQLite is reserved for isolated tests.")
    values = config.model_dump(mode="json")
    password = values.pop("password", None)
    values.pop("embedded_database", None)
    return {
        "schema_version": SCHEMA_VERSION,
        "backend": "sqlite" if config.embedded_database else "postgres",
        "connection": values,
        "password_protected": _protect_secret(password) if password else None,
    }


def write_database_config(config: DatabaseConfig, storage_root: Path) -> Path:
    path = database_config_path(storage_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(_payload(config), indent=2, sort_keys=True) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(prefix=".database-", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        if os.name != "nt":
            os.chmod(temporary, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        return path
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def read_database_config(storage_root: Path) -> DatabaseConfig:
    path = database_config_path(storage_root)
    if os.name != "nt" and path.exists() and path.stat().st_mode & 0o077:
        raise PermissionError("Database configuration permissions must be restricted to the current user.")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or set(payload) != {
            "schema_version", "backend", "connection", "password_protected"
        }:
            raise ValueError("Database configuration has an unsupported shape.")
        if payload["schema_version"] != SCHEMA_VERSION:
            raise ValueError("Database configuration schema version is unsupported.")
        if payload["backend"] not in {"sqlite", "postgres"} or not isinstance(payload["connection"], dict):
            raise ValueError("Database configuration backend is invalid.")
        values = dict(payload["connection"])
        values["embedded_database"] = payload["backend"] == "sqlite"
        protected = payload["password_protected"]
        values["password"] = _unprotect_secret(protected) if protected else None
        config = DatabaseConfig.model_validate(values)
        if config.embedded_database and config.sqlite_path == ":memory:":
            raise ValueError("In-memory SQLite is reserved for isolated tests.")
        return config
    except FileNotFoundError:
        raise
    except Exception as exc:
        raise ValueError("Database configuration is invalid or cannot be decrypted; repair database.json before startup.") from exc


__all__ = ["CONFIG_FILENAME", "database_config_path", "read_database_config", "write_database_config"]
