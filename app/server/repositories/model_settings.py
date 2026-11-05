# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select

from server.repositories.database.manager import DatabaseManager
from server.repositories.schemas.models import AdsorptionModelSetting

###############################################################################
@dataclass(frozen=True)
class StoredModelSetting:
    enabled: bool = True
    parameter_bounds: dict[str, dict[str, float]] = field(default_factory=dict)

###############################################################################
class ModelSettingsRepository:
    """Persist user-configured global defaults; absent rows mean canonical defaults."""

    # -------------------------------------------------------------------------
    def __init__(self, database: DatabaseManager) -> None:
        self.database = database

    # -------------------------------------------------------------------------
    def all(self) -> dict[str, StoredModelSetting]:
        with self.database.session_factory() as session:
            rows = session.scalars(select(AdsorptionModelSetting)).all()
            return {
                row.model_key: StoredModelSetting(
                    enabled=bool(row.enabled),
                    parameter_bounds=dict(row.parameter_bounds or {}),
                )
                for row in rows
            }

    # -------------------------------------------------------------------------
    def save(
        self,
        model_key: str,
        *,
        enabled: bool,
        parameter_bounds: dict[str, dict[str, Any]],
    ) -> None:
        with self.database.transaction() as session:
            row = session.get(AdsorptionModelSetting, model_key)
            if row is None:
                row = AdsorptionModelSetting(model_key=model_key)
                session.add(row)
            row.enabled = enabled
            row.parameter_bounds = parameter_bounds
