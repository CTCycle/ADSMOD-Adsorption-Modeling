from __future__ import annotations

import os
import shutil
import time
from dataclasses import dataclass
from typing import Any

import psutil

from ADSMOD.server.learning.training.manager import run_training_process
from ADSMOD.server.learning.training.worker import ProcessWorker


###############################################################################
@dataclass
class TrainingRunResult:
    runtime_seconds: float
    peak_rss_bytes: int
    base_rss_bytes: int
    timed_out: bool
    error: str | None
    result_payload: dict[str, Any] | None
    exit_code: int | None


# -------------------------------------------------------------------------
def list_checkpoint_folders(root: str) -> set[str]:
    if not os.path.exists(root):
        return set()
    folders: set[str] = set()
    for entry in os.scandir(root):
        if entry.is_dir():
            folders.add(entry.name)
    return folders


# -------------------------------------------------------------------------
def remove_checkpoint_folders(root: str, folders: set[str]) -> None:
    for folder in folders:
        path = os.path.join(root, folder)
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)


# -------------------------------------------------------------------------
def run_training_with_metrics(
    configuration: dict[str, Any],
    timeout_seconds: float,
    poll_interval_seconds: float,
    stop_grace_seconds: float,
    baseline_epoch: int,
) -> TrainingRunResult:
    worker = ProcessWorker()
    worker.start(target=run_training_process, kwargs={"configuration": configuration})

    start_time = time.monotonic()
    stop_requested_at: float | None = None
    timed_out = False
    base_rss = 0
    first_rss = 0
    peak_rss = 0
    process = None

    if worker.process is not None and worker.process.pid is not None:
        process = psutil.Process(worker.process.pid)

    while True:
        if process is not None:
            try:
                rss = int(process.memory_info().rss)
                if first_rss == 0:
                    first_rss = rss
                if rss > peak_rss:
                    peak_rss = rss
            except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
                process = None

        message = worker.poll(timeout=0.0)
        if (
            message
            and isinstance(message, dict)
            and message.get("type") == "epoch_end"
            and base_rss == 0
        ):
            epoch_value = message.get("epoch")
            if isinstance(epoch_value, int) and epoch_value >= baseline_epoch:
                if process is not None:
                    try:
                        base_rss = int(process.memory_info().rss)
                    except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
                        base_rss = 0

        if not worker.is_alive():
            break

        elapsed = time.monotonic() - start_time
        if elapsed >= timeout_seconds:
            if stop_requested_at is None:
                worker.stop()
                stop_requested_at = time.monotonic()
                timed_out = True
            elif time.monotonic() - stop_requested_at >= stop_grace_seconds:
                worker.terminate()
                break

        time.sleep(poll_interval_seconds)

    worker.join(timeout=5.0)

    if worker.is_alive():
        worker.terminate()
        worker.join(timeout=5.0)

    result_payload = worker.read_result()
    exit_code = worker.exitcode
    worker.cleanup()

    error = None
    if isinstance(result_payload, dict) and result_payload.get("error"):
        error = str(result_payload.get("error"))
    elif result_payload is None and exit_code not in (0, None):
        error = f"Process exited with code {exit_code}"

    runtime_seconds = time.monotonic() - start_time
    if base_rss == 0:
        base_rss = first_rss
    if peak_rss == 0 and base_rss > 0:
        peak_rss = base_rss

    return TrainingRunResult(
        runtime_seconds=runtime_seconds,
        peak_rss_bytes=peak_rss,
        base_rss_bytes=base_rss,
        timed_out=timed_out,
        error=error,
        result_payload=result_payload,
        exit_code=exit_code,
    )
