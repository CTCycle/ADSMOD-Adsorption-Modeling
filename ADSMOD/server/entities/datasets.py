from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from ADSMOD.server.common.constants import DEFAULT_DATASET_COLUMN_MAPPING
from ADSMOD.server.entities.fitting import DatasetPayload


###############################################################################
@dataclass
class DatasetColumns:
    experiment: str = DEFAULT_DATASET_COLUMN_MAPPING["experiment"]
    temperature: str = DEFAULT_DATASET_COLUMN_MAPPING["temperature"]
    pressure: str = DEFAULT_DATASET_COLUMN_MAPPING["pressure"]
    uptake: str = DEFAULT_DATASET_COLUMN_MAPPING["uptake"]

    # -------------------------------------------------------------------------
    def as_dict(self) -> dict[str, str]:
        return {
            "experiment": self.experiment,
            "temperature": self.temperature,
            "pressure": self.pressure,
            "uptake": self.uptake,
        }


###############################################################################
class DatasetLoadResponse(BaseModel):
    status: str = Field(default="success")
    summary: str
    dataset: DatasetPayload | dict[str, Any] | None = None


###############################################################################
class DatasetNamesResponse(BaseModel):
    names: list[str] = Field(default_factory=list)
