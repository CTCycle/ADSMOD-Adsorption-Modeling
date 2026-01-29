from __future__ import annotations

import multiprocessing
import os
import queue
import signal
import subprocess
from dataclasses import dataclass
from typing import Any, Callable

from ADSMOD.server.utils.learning.callbacks import WorkerInterrupted
from ADSMOD.server.utils.logger import logger


def _safe_queue_put(target_queue: multiprocessing.Queue, payload: dict[str, Any]) -> None:
    try:
        target_queue.put_nowait(payload)
    except Exception:
        target_queue.put(payload)


@dataclass
class WorkerHandle:
    stop_event: multiprocessing.Event | None
    message_queue: multiprocessing.Queue | None

    def is_interrupted(self) -> bool:
        return bool(self.stop_event and self.stop_event.is_set())

    def send_message(self, payload: dict[str, Any]) -> None:
        if self.message_queue is None:
            return
        try:
            self.message_queue.put_nowait(payload)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to send worker message: %s", exc)
            try:
                self.message_queue.put(payload)
            except Exception:
                return


def process_target(
    result_queue: multiprocessing.Queue,
    message_queue: multiprocessing.Queue,
    stop_event: multiprocessing.Event,
    runner: Callable[..., dict[str, Any]],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> None:
    if stop_event.is_set():
        _safe_queue_put(result_queue, {"status": "interrupted"})
        return

    if os.name != "nt":
        try:
            os.setsid()
        except OSError:
            pass

    worker = WorkerHandle(stop_event=stop_event, message_queue=message_queue)
    run_kwargs = dict(kwargs)
    run_kwargs.setdefault("worker", worker)
    try:
        result = runner(*args, **run_kwargs)
        _safe_queue_put(result_queue, {"status": "success", "result": result})
    except WorkerInterrupted:
        _safe_queue_put(result_queue, {"status": "interrupted"})
    except Exception as exc:  # noqa: BLE001
        error_msg = str(exc).split("\n")[0][:200]
        _safe_queue_put(result_queue, {"status": "error", "error": error_msg})
        try:
            message_queue.put_nowait({"type": "error", "error": error_msg})
        except Exception:
            pass


###############################################################################
class ProcessWorker:
    def __init__(
        self,
        target: Callable[..., dict[str, Any]],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
        stop_timeout_seconds: float = 10.0,
    ) -> None:
        self._context = multiprocessing.get_context("spawn")
        self._stop_event = self._context.Event()
        self._result_queue: multiprocessing.Queue = self._context.Queue(maxsize=1)
        self._message_queue: multiprocessing.Queue = self._context.Queue()
        self._process: multiprocessing.Process | None = None
        self._target = target
        self._args = args
        self._kwargs = kwargs or {}
        self._stop_timeout_seconds = float(stop_timeout_seconds)

    @property
    def stop_timeout_seconds(self) -> float:
        return self._stop_timeout_seconds

    # -------------------------------------------------------------------------
    def start(self) -> None:
        if self._process is not None and self._process.is_alive():
            raise RuntimeError("Process worker is already running.")

        self._process = self._context.Process(
            target=process_target,
            args=(
                self._result_queue,
                self._message_queue,
                self._stop_event,
                self._target,
                self._args,
                self._kwargs,
            ),
        )
        self._process.daemon = True
        self._process.start()

    # -------------------------------------------------------------------------
    def is_alive(self) -> bool:
        return bool(self._process and self._process.is_alive())

    # -------------------------------------------------------------------------
    def stop(self) -> None:
        self._stop_event.set()

    # -------------------------------------------------------------------------
    def interrupt(self) -> None:
        self.stop()

    # -------------------------------------------------------------------------
    def is_interrupted(self) -> bool:
        return self._stop_event.is_set()

    # -------------------------------------------------------------------------
    def poll(self, max_messages: int | None = None) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if self._message_queue is None:
            return messages

        count = 0
        while True:
            if max_messages is not None and count >= max_messages:
                break
            try:
                message = self._message_queue.get_nowait()
            except queue.Empty:
                break
            except (EOFError, OSError) as exc:
                logger.debug("Process message queue closed: %s", exc)
                break
            else:
                if isinstance(message, dict):
                    messages.append(message)
                else:
                    messages.append({"type": "message", "payload": message})
                count += 1

        return messages

    # -------------------------------------------------------------------------
    def collect_result(self) -> tuple[str | None, dict[str, Any] | None, str | None]:
        status: str | None = None
        result: dict[str, Any] | None = None
        error: str | None = None

        payload: Any | None = None
        try:
            payload = self._result_queue.get_nowait()
        except queue.Empty:
            payload = None
        except (EOFError, OSError) as exc:
            logger.debug("Process result queue closed: %s", exc)

        if isinstance(payload, dict):
            status = payload.get("status")
            if status == "error":
                error = payload.get("error") or "Process failed."
            elif status == "success":
                result = payload.get("result")
            elif status == "interrupted":
                status = "interrupted"
            else:
                result = payload
        elif payload is not None:
            status = "success"
            result = {"result": payload}

        if payload is None and self._process is not None:
            exit_code = self._process.exitcode
            if exit_code not in (None, 0):
                status = "error"
                error = f"Process exited with code {exit_code}"

        return status, result, error

    # -------------------------------------------------------------------------
    def join(self, timeout: float | None = None) -> bool:
        if self._process is None:
            return True
        self._process.join(timeout=timeout)
        return not self._process.is_alive()

    # -------------------------------------------------------------------------
    def terminate(self) -> None:
        if self._process is None or not self._process.is_alive():
            return
        self.stop()
        self._terminate_process_tree(self._process.pid)

    # -------------------------------------------------------------------------
    def cleanup(self) -> None:
        if self._process is not None:
            if self._process.is_alive():
                self.terminate()
            self._process.join(timeout=1.0)

        if self._result_queue is not None:
            self._result_queue.close()
            self._result_queue.join_thread()

        if self._message_queue is not None:
            self._message_queue.close()
            self._message_queue.join_thread()

        self._process = None

    # -------------------------------------------------------------------------
    @staticmethod
    def _terminate_process_tree(process_id: int | None) -> None:
        if process_id is None:
            return

        if os.name == "nt":
            try:
                subprocess.run(
                    ["taskkill", "/PID", str(process_id), "/T", "/F"],
                    check=False,
                    capture_output=True,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to terminate process %s: %s", process_id, exc)
            return

        try:
            os.killpg(process_id, signal.SIGTERM)
        except ProcessLookupError:
            return
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to terminate process %s: %s", process_id, exc)
            return

        try:
            os.killpg(process_id, signal.SIGKILL)
        except ProcessLookupError:
            return
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to force kill process %s: %s", process_id, exc)
