"""
GPU and CPU memory profiling utilities.

Provides point-in-time snapshots and continuous background monitoring of
device memory, with automatic forwarding to the active WSTracker run.

Classes:
    MemorySnapshot: Immutable record of GPU/CPU memory at one instant.
    MemoryTracker:  Background-thread monitor that records samples at a
                    configurable interval.
"""

from __future__ import annotations

import contextlib
import os
import threading
import time
from typing import Any, Dict, List, Optional

import torch


class MemorySnapshot:
    """Immutable record of GPU and CPU memory at a single point in time.

    All values are in megabytes (MiB).

    Attributes:
        tag: Human-readable label for this snapshot.
        gpu_allocated_mb: Bytes currently allocated by PyTorch tensors on the
            selected CUDA device, converted to MiB.
        gpu_reserved_mb: Bytes reserved (cached) by the CUDA allocator, MiB.
        gpu_peak_mb: Peak ``gpu_allocated_mb`` since the last
            ``torch.cuda.reset_peak_memory_stats()`` call, MiB.
        cpu_rss_mb: Resident Set Size of this process (requires *psutil*),
            MiB.  Zero when *psutil* is not installed.
        timestamp: ``time.time()`` at capture.
    """

    def __init__(
        self,
        tag: str,
        gpu_allocated_mb: float = 0.0,
        gpu_reserved_mb: float = 0.0,
        gpu_peak_mb: float = 0.0,
        cpu_rss_mb: float = 0.0,
        timestamp: Optional[float] = None,
    ) -> None:
        self.tag = tag
        self.gpu_allocated_mb = gpu_allocated_mb
        self.gpu_reserved_mb = gpu_reserved_mb
        self.gpu_peak_mb = gpu_peak_mb
        self.cpu_rss_mb = cpu_rss_mb
        self.timestamp = timestamp if timestamp is not None else time.time()

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def capture(
        cls,
        tag: str,
        device: Optional[torch.device] = None,
        step: int = 0,
        log_metric: bool = True,
    ) -> "MemorySnapshot":
        """Capture the current memory state and return a :class:`MemorySnapshot`.

        Args:
            tag: Label used both as the snapshot identifier and as the metric
                namespace when logging (``memory/{tag}/...``).
            device: CUDA device to query.  When ``None``, the current default
                device is used.  CPU-only environments are handled gracefully.
            step: Metric step forwarded to ``Run.log_metrics``.
            log_metric: When ``True`` (default), call :meth:`log_to_run`
                immediately after capturing.

        Returns:
            A :class:`MemorySnapshot` populated with current readings.
        """
        gpu_allocated = 0.0
        gpu_reserved = 0.0
        gpu_peak = 0.0

        if torch.cuda.is_available():
            if device is not None and device.type == "cuda":
                idx = device.index if device.index is not None else torch.cuda.current_device()
            else:
                idx = torch.cuda.current_device()
            gpu_allocated = torch.cuda.memory_allocated(idx) / 1024**2
            gpu_reserved = torch.cuda.memory_reserved(idx) / 1024**2
            gpu_peak = torch.cuda.max_memory_allocated(idx) / 1024**2

        cpu_rss = 0.0
        try:
            import psutil  # type: ignore[import-not-found]

            cpu_rss = psutil.Process(os.getpid()).memory_info().rss / 1024**2
        except ImportError:
            pass

        snap = cls(
            tag=tag,
            gpu_allocated_mb=gpu_allocated,
            gpu_reserved_mb=gpu_reserved,
            gpu_peak_mb=gpu_peak,
            cpu_rss_mb=cpu_rss,
        )

        if log_metric:
            snap.log_to_run(step=step)

        return snap

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def delta(self, baseline: "MemorySnapshot") -> Dict[str, float]:
        """Compute memory differences relative to *baseline*.

        Returns:
            Dict with keys ``gpu_allocated_delta_mb``,
            ``gpu_reserved_delta_mb``, ``gpu_peak_delta_mb``,
            ``cpu_rss_delta_mb``.
        """
        return {
            "gpu_allocated_delta_mb": self.gpu_allocated_mb - baseline.gpu_allocated_mb,
            "gpu_reserved_delta_mb": self.gpu_reserved_mb - baseline.gpu_reserved_mb,
            "gpu_peak_delta_mb": self.gpu_peak_mb - baseline.gpu_peak_mb,
            "cpu_rss_delta_mb": self.cpu_rss_mb - baseline.cpu_rss_mb,
        }

    # ------------------------------------------------------------------
    # WSTracker integration
    # ------------------------------------------------------------------

    def log_to_run(self, step: int = 0) -> None:
        """Log all memory fields to the active WSTracker run.

        Metrics are stored under the namespace ``memory/{tag}/``.
        This is a silent no-op when no run is active or any error occurs.
        """
        with contextlib.suppress(Exception):
            from tracking.core.run import get_active_run

            run = get_active_run()
            if run is not None:
                run.log_metrics(
                    {
                        f"memory/{self.tag}/gpu_allocated_mb": self.gpu_allocated_mb,
                        f"memory/{self.tag}/gpu_reserved_mb": self.gpu_reserved_mb,
                        f"memory/{self.tag}/gpu_peak_mb": self.gpu_peak_mb,
                        f"memory/{self.tag}/cpu_rss_mb": self.cpu_rss_mb,
                    },
                    step=step,
                )

    def __repr__(self) -> str:
        return (
            f"MemorySnapshot(tag={self.tag!r}, "
            f"gpu={self.gpu_allocated_mb:.1f}/{self.gpu_reserved_mb:.1f} MB alloc/reserved, "
            f"peak={self.gpu_peak_mb:.1f} MB, cpu_rss={self.cpu_rss_mb:.1f} MB)"
        )


class MemoryTracker:
    """Continuously monitors memory usage in a daemon background thread.

    Samples are collected at *interval_sec* intervals.  Use as a context
    manager or call :meth:`start` / :meth:`stop` manually.  Call
    :meth:`log_summary_to_run` once after stopping to forward the peak values
    to the active WSTracker run.

    Usage::

        tracker = MemoryTracker(interval_sec=0.5, tag="training")
        with tracker:
            train_one_epoch()
        print(tracker.peak_gpu_mb)
        tracker.log_summary_to_run(step=epoch)

    Args:
        interval_sec: Sampling interval in seconds.
        tag: Metric namespace label; forwarded to each
            :class:`MemorySnapshot`.
        device: CUDA device to query (``None`` = default device).
        log_per_sample: When ``True``, each snapshot is also logged to the
            active run as a time-series metric.  Defaults to ``False`` to
            avoid metric spam; use :meth:`log_summary_to_run` instead.
    """

    def __init__(
        self,
        interval_sec: float = 1.0,
        tag: str = "background",
        device: Optional[torch.device] = None,
        log_per_sample: bool = False,
    ) -> None:
        self.interval_sec = interval_sec
        self.tag = tag
        self.device = device
        self.log_per_sample = log_per_sample

        self._samples: List[Dict[str, float]] = []
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
    # Aggregate properties
    # ------------------------------------------------------------------

    @property
    def peak_gpu_mb(self) -> float:
        """Peak GPU-allocated memory observed across all samples (MiB)."""
        if not self._samples:
            return 0.0
        return max(s["gpu_allocated_mb"] for s in self._samples)

    @property
    def peak_cpu_mb(self) -> float:
        """Peak CPU RSS observed across all samples (MiB)."""
        if not self._samples:
            return 0.0
        return max(s["cpu_rss_mb"] for s in self._samples)

    @property
    def n_samples(self) -> int:
        """Number of snapshots collected since the last :meth:`start`."""
        return len(self._samples)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> "MemoryTracker":
        """Start the background monitoring thread.  Returns *self*."""
        self._samples.clear()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitor, daemon=True, name=f"MemoryTracker-{self.tag}")
        self._thread.start()
        return self

    def stop(self) -> "MemoryTracker":
        """Signal the monitoring thread to stop and wait for it to exit."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(self.interval_sec * 2, 2.0))
        return self

    def _monitor(self) -> None:
        while True:
            snap = MemorySnapshot.capture(
                self.tag,
                device=self.device,
                log_metric=self.log_per_sample,
            )
            self._samples.append(
                {
                    "timestamp": snap.timestamp,
                    "gpu_allocated_mb": snap.gpu_allocated_mb,
                    "gpu_reserved_mb": snap.gpu_reserved_mb,
                    "gpu_peak_mb": snap.gpu_peak_mb,
                    "cpu_rss_mb": snap.cpu_rss_mb,
                }
            )
            if self._stop_event.wait(self.interval_sec):
                break

    # ------------------------------------------------------------------
    # WSTracker integration
    # ------------------------------------------------------------------

    def log_summary_to_run(self, step: int = 0) -> None:
        """Log peak GPU and CPU memory and sample count to the active run.

        Metrics are stored under ``memory/{tag}/``.
        """
        if not self._samples:
            return
        with contextlib.suppress(Exception):
            from tracking.core.run import get_active_run

            run = get_active_run()
            if run is not None:
                run.log_metrics(
                    {
                        f"memory/{self.tag}/peak_gpu_allocated_mb": self.peak_gpu_mb,
                        f"memory/{self.tag}/peak_cpu_rss_mb": self.peak_cpu_mb,
                        f"memory/{self.tag}/n_samples": float(self.n_samples),
                    },
                    step=step,
                )

    # ------------------------------------------------------------------
    # Context-manager protocol
    # ------------------------------------------------------------------

    def __enter__(self) -> "MemoryTracker":
        return self.start()

    def __exit__(self, *_args: Any) -> None:
        self.stop()

    def __repr__(self) -> str:
        return (
            f"MemoryTracker(tag={self.tag!r}, n_samples={self.n_samples}, "
            f"peak_gpu={self.peak_gpu_mb:.1f} MB, peak_cpu={self.peak_cpu_mb:.1f} MB)"
        )
