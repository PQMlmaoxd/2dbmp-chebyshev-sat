from __future__ import annotations

import math
import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True, slots=True)
class ProcessResult:
    returncode: int
    elapsed_s: float
    timed_out: bool
    termination_signal: int | None


class _SupervisorSignal(BaseException):
    def __init__(self, signal_number: int) -> None:
        super().__init__(signal_number)
        self.signal_number = signal_number


def run_supervised(
    command: Sequence[str], *, cwd: str | Path, timeout_s: float, stdout_path: str | Path, stderr_path: str | Path
) -> ProcessResult:
    if not isinstance(timeout_s, (int, float)) or isinstance(timeout_s, bool) or not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("timeout_s must be a finite positive number")
    stdout_file, stderr_file = Path(stdout_path), Path(stderr_path)
    stdout_file.parent.mkdir(parents=True, exist_ok=True)
    stderr_file.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with stdout_file.open("w", encoding="utf-8") as stdout, stderr_file.open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(list(command), cwd=Path(cwd), stdout=stdout, stderr=stderr, start_new_session=True)
        process_group = process.pid
        previous_handlers: dict[int, object] = {}

        def handle_signal(signal_number: int, _frame: object) -> None:
            raise _SupervisorSignal(signal_number)

        if threading.current_thread() is threading.main_thread():
            for signal_number in (signal.SIGTERM, signal.SIGHUP):
                try:
                    previous_handlers[signal_number] = signal.signal(signal_number, handle_signal)
                except (ValueError, OSError):
                    pass
        try:
            process.wait(timeout=timeout_s)
            termination_signal = None
            if process.returncode != 0 or _group_exists(process_group):
                termination_signal = _stop_process_group(process, process_group)
            signal_number = -process.returncode if process.returncode < 0 else None
            return ProcessResult(
                process.returncode,
                time.perf_counter() - started,
                False,
                signal_number or termination_signal,
            )
        except subprocess.TimeoutExpired:
            signal_number = _stop_process_group(process, process_group)
            return ProcessResult(process.returncode, time.perf_counter() - started, True, signal_number)
        except _SupervisorSignal:
            _stop_process_group(process, process_group)
            raise
        except BaseException:
            _stop_process_group(process, process_group)
            raise
        finally:
            for signal_number, handler in previous_handlers.items():
                try:
                    signal.signal(signal_number, handler)
                except (ValueError, OSError):
                    pass


def _group_exists(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
    except PermissionError:
        return True
    except ProcessLookupError:
        return False
    return True


def _terminate_process_group(process: subprocess.Popen[bytes], process_group: int, termination_signal: int) -> bool:
    try:
        os.killpg(process_group, termination_signal)
        return True
    except ProcessLookupError:
        if process.poll() is None:
            process.terminate() if termination_signal == signal.SIGTERM else process.kill()
            return True
    return False


def _stop_process_group(process: subprocess.Popen[bytes], process_group: int) -> int | None:
    termination_signal: int | None = None
    if _terminate_process_group(process, process_group, signal.SIGTERM):
        termination_signal = signal.SIGTERM
    try:
        process.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        if _terminate_process_group(process, process_group, signal.SIGKILL):
            termination_signal = signal.SIGKILL
        try:
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            pass

    deadline = time.perf_counter() + 1.0
    while _group_exists(process_group) and time.perf_counter() < deadline:
        time.sleep(0.01)
    if _group_exists(process_group):
        if _terminate_process_group(process, process_group, signal.SIGKILL):
            termination_signal = signal.SIGKILL
        deadline = time.perf_counter() + 1.0
        while _group_exists(process_group) and time.perf_counter() < deadline:
            time.sleep(0.01)
    return termination_signal
