from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ADSMOD.server.entities.fitting import DatasetPayload


###############################################################################
class DatasetLoadResponse(BaseModel):
    status: str = Field(default="success")
    summary: str
    dataset: DatasetPayload | dict[str, Any] | None = None


###############################################################################
class DatasetNamesResponse(BaseModel):
    names: list[str] = Field(default_factory=list)
