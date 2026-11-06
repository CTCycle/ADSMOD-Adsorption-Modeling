# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

from pydantic import BaseModel, Field


###############################################################################
class SystemStatusResponse(BaseModel):
    cpu_percent: float = Field(ge=0.0, le=100.0)
    memory_mb: float = Field(ge=0.0)
    database_size_bytes: int | None = Field(default=None, ge=0)


__all__ = ["SystemStatusResponse"]
