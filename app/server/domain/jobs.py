# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

"""Pydantic transport/workflow contracts for background jobs."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from server.domain.public_data import PublicDataSyncRunView

###############################################################################
class JobStartResponse(BaseModel):
    job_id: str
    job_type: str
    status: str
    message: str
    poll_interval: float | None = None

###############################################################################
class JobStatusResponse(BaseModel):
    job_id: str
    job_type: str
    status: str
    progress: float
    result: dict[str, Any] | None = None
    error: str | None = None
    poll_interval: float | None = None

###############################################################################
class JobListResponse(BaseModel):
    jobs: list[JobStatusResponse]

###############################################################################
class UnifiedJobView(BaseModel):
    id: str
    kind: Literal["acquisition", "fitting", "training"]
    name: str
    source: str | None = None
    status: Literal[
        "queued",
        "running",
        "completed",
        "partially_completed",
        "failed",
        "cancelled",
        "interrupted",
    ]
    progress: float = Field(ge=0.0, le=100.0)
    created_at: datetime
    finished_at: datetime | None = None
    saved_count: int | None = None
    skipped_count: int | None = None
    failed_count: int | None = None
    error: str | None = None
    run: PublicDataSyncRunView | None = None

###############################################################################
class UnifiedJobListResponse(BaseModel):
    jobs: list[UnifiedJobView]
    active_count: int = Field(ge=0)

###############################################################################
class JobCancelResponse(BaseModel):
    status: str
    job_id: str

###############################################################################
class StatusMessageResponse(BaseModel):
    status: str
    message: str
