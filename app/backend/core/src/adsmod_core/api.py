from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


###############################################################################
class SnapshotCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rows: list[dict[str, Any]] = Field(min_length=1)


###############################################################################
class SnapshotCreateResponse(BaseModel):
    snapshot_id: str
    content_hash: str
    created_at: str
    row_count: int


###############################################################################
class SnapshotPageResponse(BaseModel):
    snapshot_id: str
    content_hash: str
    page: int
    page_size: int
    total_rows: int
    rows: list[dict[str, Any]]