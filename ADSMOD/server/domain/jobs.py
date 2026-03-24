"""Pydantic schemas for background job responses."""

from __future__ import annotations

import multiprocessing
import threading
from dataclasses import dataclass, field
from time import monotonic
from typing import Any

from collections.abc import Callable

from pydantic import BaseModel


###############################################################################
@dataclass
class JobState:
    job_id: str
    job_type: str
    status: str
    progress: float = 0.0
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: float = field(default_factory=monotonic)
    completed_at: float | None = None
    stop_requested: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    # -------------------------------------------------------------------------
    def update(self, **kwargs: Any) -> None:
        with self.lock:
            for key, value in kwargs.items():
                if hasattr(self, key):
                    setattr(self, key, value)

    # -------------------------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "job_id": self.job_id,
                "job_type": self.job_type,
                "status": self.status,
                "progress": self.progress,
                "result": self.result,
                "error": self.error,
                "created_at": self.created_at,
                "completed_at": self.completed_at,
            }


###############################################################################
@dataclass
class JobExecutionConfig:
    run_mode: str
    process_stop_timeout_seconds: float
    process_message_handler: Callable[[str, dict[str, Any]], None] | None = None
    completion_handler: (
        Callable[[str, str, dict[str, Any] | None, str | None], None] | None
    ) = None


###############################################################################
@dataclass
class ProcessJobState:
    process: multiprocessing.Process
    stop_event: multiprocessing.Event
    result_queue: multiprocessing.Queue
    message_queue: multiprocessing.Queue
    created_at: float = field(default_factory=monotonic)


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
