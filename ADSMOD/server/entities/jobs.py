"""Pydantic schemas for background job responses."""

from __future__ import annotations

from pydantic import BaseModel


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
    result: dict | None = None
    error: str | None = None
    poll_interval: float | None = None


###############################################################################
class JobListResponse(BaseModel):
    jobs: list[JobStatusResponse]
