from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from typing import ClassVar

from pydantic import ValidationError

from ADSMOD.server.common.constants import CONFIGURATION_FILE
from ADSMOD.server.domain.settings import AppSettings, ServerSettings


###############################################################################
class ConfigurationManager:
    def __init__(self, config_path: str | None = None) -> None:
        self.config_path = config_path or CONFIGURATION_FILE
        self.settings: AppSettings | None = None

    # -------------------------------------------------------------------------
    def load(self) -> AppSettings:
        settings_cls = self._build_path_scoped_settings_class(self.config_path)
        try:
            self.settings = settings_cls.load()
        except ValidationError as exc:
            raise RuntimeError(f"Invalid application settings: {exc}") from exc
        return self.settings

    # -------------------------------------------------------------------------
    def reload(self) -> AppSettings:
        return self.load()

    # -------------------------------------------------------------------------
    def update(self, payload: dict[str, Any]) -> AppSettings:
        current = self.load_configuration_data(self.config_path)
        updated = self._ensure_mapping(current)
        updated.update(self._ensure_mapping(payload))
        with open(self.config_path, "w", encoding="utf-8") as handle:
            json.dump(updated, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        return self.reload()

    # -------------------------------------------------------------------------
    def get_block(self, section: str) -> dict[str, Any]:
        loaded = self.load_configuration_data(self.config_path)
        block = loaded.get(section, {})
        return self._ensure_mapping(block)

    # -------------------------------------------------------------------------
    def get_value(self, section: str, key: str, default: Any = None) -> Any:
        block = self.get_block(section)
        return block.get(key, default)

    # -------------------------------------------------------------------------
    def to_server_settings(self) -> ServerSettings:
        if self.settings is None:
            self.load()
        if self.settings is None:
            raise RuntimeError("Application settings are not available.")
        return self.settings.to_server_settings()

    # -------------------------------------------------------------------------
    @staticmethod
    def _build_path_scoped_settings_class(config_path: str) -> type[AppSettings]:
        class PathScopedAppSettings(AppSettings):
            _configuration_file: ClassVar[str] = config_path

        return PathScopedAppSettings

    # -------------------------------------------------------------------------
    @staticmethod
    def _ensure_mapping(value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return value
        return {}

    # -------------------------------------------------------------------------
    @staticmethod
    def load_configuration_data(path: str | None = None) -> dict[str, Any]:
        resolved_path = Path(path or CONFIGURATION_FILE)
        if not resolved_path.exists():
            raise RuntimeError(f"Configuration file not found: {resolved_path}")
        try:
            with open(resolved_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Unable to load configuration from {resolved_path}"
            ) from exc
        if not isinstance(data, dict):
            raise RuntimeError("Configuration must be a JSON object.")
        return data


