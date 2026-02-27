"""Run lifecycle management for the WSTracker."""

from __future__ import annotations

import contextlib
import math
import os
import threading
from typing import Any, Dict, List, Optional, Tuple

from ..validation.hashing import hash_file
from .store import TrackingStore

# ---------------------------------------------------------------------------
# Module-level active-run registry (one per process)
# ---------------------------------------------------------------------------

_active_run: Optional["Run"] = None
_registry_lock = threading.Lock()


def get_active_run() -> Optional["Run"]:
    """Return the active :class:`Run` for the current process, or ``None``."""
    return _active_run


def set_active_run(run: Optional["Run"]) -> None:
    """Set the active :class:`Run` for the current process."""
    global _active_run
    with _registry_lock:
        _active_run = run


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _flatten_dict(d: Dict[str, Any], prefix: str = "", sep: str = ".") -> Dict[str, Any]:
    """Recursively flatten a nested dict using dot-separated keys."""
    result: Dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{sep}{k}" if prefix else k
        if isinstance(v, dict):
            result.update(_flatten_dict(v, key, sep))
        else:
            result[key] = v
    return result


def _safe_float(v: Any) -> Optional[float]:
    try:
        fval = float(v)
        if math.isnan(fval) or math.isinf(fval):
            return None
        return fval
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


class Run:
    """A single tracked experiment run.

    Provides methods to log metrics, parameters, tags, artifacts, and
    dataset lifecycle events.  All metric writes are buffered in memory
    and flushed to SQLite in batches for efficiency.

    This class is **thread-safe**: multiple threads in the same process
    may call :meth:`log_metric` / :meth:`log_metrics` concurrently.

    Args:
        run_id: UUID string identifying the run.
        store: Backend storage instance.
        artifact_dir: Root directory for artifact files.
        buffer_size: Number of metric entries to buffer before a flush.
    """

    def __init__(
        self,
        run_id: str,
        store: TrackingStore,
        artifact_dir: str,
        buffer_size: int = 200,
    ) -> None:
        self.run_id = run_id
        self._store = store
        self.artifact_dir = artifact_dir
        self._buffer_size = buffer_size

        self._metric_buffer: List[Tuple[str, float, int]] = []
        self._buffer_lock = threading.Lock()

        # path -> last observed SHA-256 for change detection
        self._file_hashes: Dict[str, str] = {}
        self._closed = False

        # Optional secondary logging sinks (e.g. MLflowBridge).
        # Each sink must expose log_metric(key, value, step),
        # log_params(params), log_artifact(path), and finish(status).
        self._sinks: List[Any] = []

    # ------------------------------------------------------------------
    # Tags
    # ------------------------------------------------------------------

    def add_sink(self, sink: Any) -> "Run":
        """Attach a secondary logging sink to this run.

        The sink receives forwarded calls for every metric, param, and
        artifact logged hereafter.  It must implement:

        * ``log_metric(key: str, value: float, step: int)``
        * ``log_params(params: dict)``
        * ``log_artifact(path: str)``
        * ``finish(status: str)``

        Errors raised by the sink are silently suppressed so a broken
        secondary backend never disrupts the primary WSTracker writes.

        Args:
            sink: Any object implementing the sink protocol above.

        Returns:
            ``self`` for method chaining.
        """
        self._sinks.append(sink)
        return self

    def set_tag(self, key: str, value: str) -> "Run":
        """Attach a metadata tag to this run."""
        if not self._closed:
            self._store.set_tag(self.run_id, key, str(value))
        return self

    def set_tags(self, tags: Dict[str, str]) -> "Run":
        """Attach multiple metadata tags to this run."""
        if not self._closed:
            self._store.set_tags(self.run_id, {k: str(v) for k, v in tags.items()})
        return self

    # ------------------------------------------------------------------
    # Params
    # ------------------------------------------------------------------

    def log_param(self, key: str, value: Any) -> "Run":
        """Record a single configuration parameter (set-once semantics)."""
        if not self._closed:
            self._store.log_param(self.run_id, key, value)
        return self

    def log_params(self, params: Dict[str, Any]) -> "Run":
        """Record multiple configuration parameters, flattening nested dicts."""
        if not self._closed:
            flat = _flatten_dict(params)
            self._store.log_params(self.run_id, flat)
            for sink in self._sinks:
                with contextlib.suppress(Exception):
                    sink.log_params(flat)
        return self

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def log_metric(self, key: str, value: Any, step: int = 0) -> "Run":
        """Log a single scalar metric value at the given step."""
        if self._closed:
            return self
        fval = _safe_float(value)
        if fval is None:
            return self
        with self._buffer_lock:
            self._metric_buffer.append((key, fval, step))
            if len(self._metric_buffer) >= self._buffer_size:
                self._flush_metrics_locked()
        for sink in self._sinks:
            with contextlib.suppress(Exception):
                sink.log_metric(key, fval, step)
        return self

    def log_metrics(self, metrics: Dict[str, Any], step: int = 0) -> "Run":
        """Log a dict of scalar metric values at the given step."""
        if self._closed:
            return self
        with self._buffer_lock:
            for k, v in metrics.items():
                fval = _safe_float(v)
                if fval is not None:
                    self._metric_buffer.append((k, fval, step))
            if len(self._metric_buffer) >= self._buffer_size:
                self._flush_metrics_locked()
        return self

    def flush(self) -> "Run":
        """Force-flush any buffered metrics to the store."""
        with self._buffer_lock:
            self._flush_metrics_locked()
        return self

    def _flush_metrics_locked(self) -> None:
        """Flush buffer — caller must hold ``_buffer_lock``."""
        if self._metric_buffer:
            self._store.log_metrics_batch(self.run_id, self._metric_buffer)
            self._metric_buffer.clear()

    # ------------------------------------------------------------------
    # Artifacts
    # ------------------------------------------------------------------

    def log_artifact(
        self,
        path: str,
        name: Optional[str] = None,
        artifact_type: str = "file",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "Run":
        """Register a file as an artifact of this run.

        Args:
            path: Path to the file.
            name: Human-readable name (defaults to basename of *path*).
            artifact_type: Semantic type, e.g. ``'model'``, ``'checkpoint'``,
                ``'config'``, ``'result'``.
            metadata: Arbitrary extra metadata dict.
        """
        if self._closed or not os.path.exists(path):
            return self
        self._store.log_artifact(
            self.run_id,
            name or os.path.basename(path),
            path,
            artifact_type=artifact_type,
            file_hash=hash_file(path),
            size_bytes=os.path.getsize(path),
            metadata=metadata,
        )
        for sink in self._sinks:
            with contextlib.suppress(Exception):
                sink.log_artifact(path)
        return self

    def log_artifacts_dir(
        self,
        dir_path: str,
        artifact_type: str = "file",
    ) -> "Run":
        """Register every file in *dir_path* as an artifact."""
        if not os.path.isdir(dir_path):
            return self
        for fname in os.listdir(dir_path):
            fpath = os.path.join(dir_path, fname)
            if os.path.isfile(fpath):
                self.log_artifact(fpath, artifact_type=artifact_type)
        return self

    # ------------------------------------------------------------------
    # Dataset events
    # ------------------------------------------------------------------

    def log_dataset_event(
        self,
        event_type: str,
        file_path: Optional[str] = None,
        shape: Optional[tuple] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "Run":
        """Record a data-lifecycle event.

        Args:
            event_type: One of ``'load'``, ``'generate'``, ``'mutate'``,
                ``'save'``, ``'hash_change'``, ``'schema_change'``.
            file_path: Path to the data file (optional).
            shape: Shape of the data item (optional).
            metadata: Arbitrary extra context dict.
        """
        if self._closed:
            return self

        file_hash: Optional[str] = None
        prev_hash: Optional[str] = None
        size_bytes: Optional[int] = None

        if file_path and os.path.exists(file_path):
            file_hash = hash_file(file_path)
            size_bytes = os.path.getsize(file_path)
            prev_hash = self._file_hashes.get(file_path)
            self._file_hashes[file_path] = file_hash  # type: ignore[assignment]

        self._store.log_dataset_event(
            self.run_id,
            event_type=event_type,
            file_path=file_path,
            file_hash=file_hash,
            prev_hash=prev_hash,
            size_bytes=size_bytes,
            shape=shape,
            metadata=metadata,
        )
        return self

    def watch_file(self, file_path: str) -> "Run":
        """Register the current hash of *file_path* for future change detection."""
        if os.path.exists(file_path):
            h = hash_file(file_path)
            if h is not None:
                self._file_hashes[file_path] = h
        return self

    def check_file_changed(self, file_path: str) -> bool:
        """Return ``True`` and record an event if *file_path* has changed.

        The internal hash map is updated on every call regardless of the result,
        so the next call uses the latest version as the baseline.
        """
        if not os.path.exists(file_path):
            return False
        current = hash_file(file_path)
        prev = self._file_hashes.get(file_path)
        if current is not None:
            self._file_hashes[file_path] = current
        changed = prev is None or current != prev
        if changed and prev is not None:
            # Observed change — log it automatically
            self._store.log_dataset_event(
                self.run_id,
                event_type="hash_change",
                file_path=file_path,
                file_hash=current,
                prev_hash=prev,
                size_bytes=os.path.getsize(file_path),
            )
        return changed

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def finish(self, status: str = "completed", error: Optional[str] = None) -> None:
        """Flush buffers and mark this run as finished in the store."""
        if not self._closed:
            self.flush()
            self._store.finish_run(self.run_id, status=status, error=error)
            self._closed = True
            for sink in self._sinks:
                with contextlib.suppress(Exception):
                    sink.finish(status)
            if get_active_run() is self:
                set_active_run(None)

    # Context-manager support
    def __enter__(self) -> "Run":
        set_active_run(self)
        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_val: Any,
        exc_tb: Any,
    ) -> bool:
        if exc_type is not None:
            self.finish(status="failed", error=str(exc_val))
        else:
            self.finish(status="completed")
        return False  # do not suppress exceptions

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_params(self) -> Dict[str, Any]:
        """Return all logged parameters."""
        return self._store.get_params(self.run_id)

    def get_tags(self) -> Dict[str, str]:
        """Return all logged tags."""
        return self._store.get_tags(self.run_id)

    def get_latest_metrics(self) -> Dict[str, float]:
        """Return the latest value for every logged metric key."""
        return self._store.get_latest_metrics(self.run_id)

    def get_metric_history(self, key: str) -> List[Dict[str, Any]]:
        """Return the full step-indexed history for a metric *key*."""
        return self._store.get_metric_history(self.run_id, key)

    def get_artifacts(self, artifact_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return registered artifacts, optionally filtered by *artifact_type*."""
        return self._store.get_artifacts(self.run_id, artifact_type=artifact_type)

    def get_dataset_events(self) -> List[Dict[str, Any]]:
        """Return all dataset lifecycle events in chronological order."""
        return self._store.get_dataset_events(self.run_id)

    def __repr__(self) -> str:
        return f"Run(id={self.run_id!r}, closed={self._closed})"
