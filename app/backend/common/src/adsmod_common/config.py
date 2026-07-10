from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

RuntimeMode = Literal["core", "core-ml"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RuntimeConfig(StrictModel):
    mode: RuntimeMode = "core"
    host: str = "127.0.0.1"
    core_port: int = Field(default=6045, ge=1024, le=65535)
    ml_port: int = Field(default=6046, ge=1024, le=65535)
    frontend_port: int = Field(default=5173, ge=1024, le=65535)
    ml_restart_attempts: int = Field(default=0, ge=0, le=1)

    @model_validator(mode="after")
    def validate_runtime(self) -> "RuntimeConfig":
        ports = {"core_port": self.core_port, "ml_port": self.ml_port, "frontend_port": self.frontend_port}
        duplicates = {port for port in ports.values() if list(ports.values()).count(port) > 1}
        if duplicates:
            names = ", ".join(name for name, port in ports.items() if port in duplicates)
            raise ValueError(f"runtime ports must be distinct: {names}")
        if self.host not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("runtime.host must be a loopback address")
        if self.mode == "core" and self.ml_restart_attempts:
            raise ValueError("runtime.ml_restart_attempts must be 0 in core mode")
        return self


class StorageConfig(StrictModel):
    root: Path = Path("%LOCALAPPDATA%/ADSMOD")
    database: str = "data/database.db"


class SecurityConfig(StrictModel):
    internal_token_required: bool = True


class AdsmodConfig(StrictModel):
    version: Literal["3.0.0"] = "3.0.0"
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)

    @model_validator(mode="after")
    def validate_mode_security(self) -> "AdsmodConfig":
        if self.runtime.mode == "core-ml" and not self.security.internal_token_required:
            raise ValueError("security.internal_token_required must be true in core-ml mode")
        return self


def load_config(path: str | Path) -> AdsmodConfig:
    config_path = Path(path)
    try:
        payload: Any = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"configuration file not found: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"configuration is not valid JSON: {config_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("configuration root must be a JSON object")
    required_sections = {"version", "runtime", "storage", "security"}
    missing_sections = required_sections.difference(payload)
    if missing_sections:
        missing = ", ".join(sorted(missing_sections))
        raise ValueError(f"configuration is missing required sections: {missing}")
    return AdsmodConfig.model_validate(payload)


__all__ = ["AdsmodConfig", "RuntimeConfig", "RuntimeMode", "SecurityConfig", "StorageConfig", "load_config"]