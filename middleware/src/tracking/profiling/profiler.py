"""
Global Execution Profiler.

Provides a ``sys.setprofile`` hook that records every function call duration
within the project to a CSV file.  Writes are accumulated in an in-memory
buffer and flushed in batches to avoid per-call file-open overhead.

Classes:
    ExecutionProfiler: Thread-safe, buffered execution profiler.

Functions:
    start_global_profiling: Start the singleton global profiler.
    stop_global_profiling: Stop the profiler and optionally log the CSV.
"""

from __future__ import annotations

import atexit
import contextlib
import datetime
import logging
import os
import sys
import threading
import time
from typing import Any, List, Optional

from constants import ROOT_DIR

logger = logging.getLogger(__name__)

_ROOT_DIR = str(ROOT_DIR)

# Write buffer: flush after this many accumulated entries.
_BUFFER_SIZE = 200

# Substrings that indicate a file should *not* be profiled.
_LIB_DIRS = (
    "site-packages",
    "dist-packages",
    "lib/python",
    "<frozen",
    "<built-in",
    # Filter the tracking directory to prevent self-instrumentation noise.
    "logic/src/tracking/",
)


class ExecutionProfiler:
    """Thread-safe, buffered function-level execution profiler.

    Uses ``sys.setprofile`` to intercept every call/return event within the
    project directory, measures wall-clock duration, and writes records to a
    CSV file.  Writes are batched in an in-memory buffer (flushed every
    :attr:`buffer_size` entries) to minimise per-call I/O overhead.

    The CSV columns are::

        timestamp,file,class,function,duration_sec

    Args:
        log_dir: Directory for the output CSV (relative to *ROOT_DIR* or
            absolute).
        buffer_size: Number of log entries to accumulate before flushing to
            disk.  A higher value reduces I/O overhead at the cost of more
            memory and potentially losing data on a crash.
    """

    def __init__(self, log_dir: str = "logs", buffer_size: int = _BUFFER_SIZE) -> None:
        self.log_dir = os.path.join(_ROOT_DIR, log_dir) if not os.path.isabs(log_dir) else log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.log_file = os.path.join(self.log_dir, f"function_execution_times_{timestamp}.csv")
        self.buffer_size = buffer_size
        self.active = False
        self._lock = threading.Lock()

        # Wall-clock start/stop for calculating coverage.
        self._wall_start: Optional[float] = None
        self._wall_stop: Optional[float] = None

        # Thread-local call stacks and start times.
        self._local = threading.local()

        # In-memory write buffer; flushed every `buffer_size` entries.
        self._write_buffer: List[str] = []

        try:
            with open(self.log_file, "w") as f:
                f.write("timestamp,file,class,function,duration_sec\n")
        except Exception as e:
            logger.error(f"Failed to initialise profiler log file: {e}")

    # ------------------------------------------------------------------
    # Filtering helpers
    # ------------------------------------------------------------------

    def _should_profile(self, filename: str) -> bool:
        """Return ``True`` when *filename* belongs to the project."""
        if not filename or filename == "<string>":
            return False
        for lib_dir in _LIB_DIRS:
            if lib_dir in filename:
                return False
        return filename.startswith(_ROOT_DIR)

    def _get_class_name(self, frame: Any) -> str:
        """Extract class name from *frame* locals (instance or class method)."""
        if "self" in frame.f_locals:
            return frame.f_locals["self"].__class__.__name__
        if "cls" in frame.f_locals:
            cls = frame.f_locals["cls"]
            if hasattr(cls, "__name__"):
                return cls.__name__
        return ""

    # ------------------------------------------------------------------
    # sys.setprofile hook
    # ------------------------------------------------------------------

    def profile_hook(self, frame: Any, event: str, arg: Any) -> None:  # noqa: ARG002
        """Profile hook installed via ``sys.setprofile``."""
        if not self.active:
            return

        if not hasattr(self._local, "call_stack"):
            self._local.call_stack = []

        code = frame.f_code
        filename = code.co_filename

        if not self._should_profile(filename):
            return

        func_name = code.co_name

        if event == "call":
            self._local.call_stack.append(
                (id(frame), time.perf_counter(), filename, self._get_class_name(frame), func_name)
            )
        elif event == "return":
            self._handle_return(frame)

    def _handle_return(self, frame: Any) -> None:
        """Process a ``return`` event: compute duration and buffer the entry."""
        frame_id = id(frame)
        stack = self._local.call_stack

        # Search backwards to support recursion and unexpected stack changes.
        match_index = -1
        for i in range(len(stack) - 1, -1, -1):
            if stack[i][0] == frame_id:
                match_index = i
                break

        if match_index == -1:
            return

        _, start_time, filename, class_name, func_name = stack[match_index]
        self._local.call_stack = stack[:match_index]

        duration = time.perf_counter() - start_time
        rel_filename = os.path.relpath(filename, _ROOT_DIR)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"{timestamp},{rel_filename},{class_name},{func_name},{duration:.6f}\n"

        with self._lock:
            self._write_buffer.append(entry)
            if len(self._write_buffer) >= self.buffer_size:
                self._flush_locked()

    def _flush_locked(self) -> None:
        """Write buffered entries to disk.  Caller must hold ``self._lock``."""
        if not self._write_buffer:
            return
        try:
            with open(self.log_file, "a") as f:
                f.writelines(self._write_buffer)
            self._write_buffer.clear()
        except Exception:
            pass

    def flush(self) -> None:
        """Flush the write buffer to disk (thread-safe)."""
        with self._lock:
            self._flush_locked()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Activate the profiler and install the global hook."""
        if not self.active:
            self.active = True
            self._wall_start = time.perf_counter()
            self._wall_stop = None
            sys.setprofile(self.profile_hook)
            threading.setprofile(self.profile_hook)
            logger.info(f"Global execution profiling started. Logging to {self.log_file}")

    def stop(self) -> None:
        """Deactivate the profiler, flush the write buffer, and remove hooks."""
        if self.active:
            self.active = False
            self._wall_stop = time.perf_counter()
            sys.setprofile(None)
            threading.setprofile(None)
            self.flush()
            logger.info("Global execution profiling stopped.")

    @property
    def wall_elapsed(self) -> Optional[float]:
        """Wall-clock seconds the profiler was active, or ``None``."""
        if self._wall_start is None:
            return None
        end = self._wall_stop if self._wall_stop is not None else time.perf_counter()
        return end - self._wall_start

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def get_report(self) -> Any:
        """Flush pending writes and return a :class:`ProfilingReport` for this run.

        Returns:
            :class:`ProfilingReport` backed by :attr:`log_file`.
        """
        self.flush()
        from .report import ProfilingReport

        return ProfilingReport(self.log_file, wall_elapsed=self.wall_elapsed)


# ---------------------------------------------------------------------------
# Module-level singleton API
# ---------------------------------------------------------------------------

_profiler_instance: Optional[ExecutionProfiler] = None


def start_global_profiling(log_dir: str = "logs", buffer_size: int = _BUFFER_SIZE) -> None:
    """Initialise and start the singleton global profiler.

    Calling this function more than once is a no-op (the existing profiler is
    reused).  An :func:`atexit` handler is registered so that the write buffer
    is always flushed before the process exits.

    Args:
        log_dir: Output directory for the CSV file (relative to *ROOT_DIR* or
            absolute).
        buffer_size: Number of log lines buffered before flushing to disk.
    """
    global _profiler_instance
    if _profiler_instance is None:
        _profiler_instance = ExecutionProfiler(log_dir=log_dir, buffer_size=buffer_size)
        _profiler_instance.start()
        atexit.register(stop_global_profiling)


def stop_global_profiling(log_artifact: bool = True, print_report: bool = True) -> None:
    """Stop the singleton profiler and optionally register the CSV as an artifact.

    Args:
        log_artifact: When ``True`` (default), register the CSV file as a
            ``"profiling"`` artifact on the active WSTracker run (if any).
        print_report: When ``True`` (default), print a summary report to
            the terminal showing aggregate statistics.
    """
    global _profiler_instance
    if _profiler_instance is not None:
        wall_elapsed = _profiler_instance.wall_elapsed
        log_file = _profiler_instance.log_file
        _profiler_instance.stop()

        report = None
        if print_report or log_artifact:
            try:
                from .report import ProfilingReport

                report = ProfilingReport(log_file, wall_elapsed=wall_elapsed)
            except Exception:
                pass

        if print_report and report is not None:
            with contextlib.suppress(Exception):
                print("\n" + str(report))

        if log_artifact:
            # Reverted: No longer logging profiling CSV as an artifact to avoid Excel row limit issues.
            pass
