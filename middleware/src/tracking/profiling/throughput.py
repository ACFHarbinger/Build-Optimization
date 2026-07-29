"""
Items-per-second throughput tracking.

Provides a sliding-window throughput meter suitable for monitoring training
step speed, simulation day rate, evaluation sample rate, and any other
iterative process where items-per-second is a key efficiency signal.

Classes:
    ThroughputTracker: Sliding-window and lifetime throughput tracker.
"""

from __future__ import annotations

import contextlib
import time
from typing import Any, Dict, Generator, List, Optional, Tuple


class ThroughputTracker:
    """Tracks items-per-second throughput over a configurable sliding window.

    Two throughput figures are maintained:

    * **window throughput** — computed over the last *window* recorded steps
      (useful for monitoring real-time training speed).
    * **total throughput** — computed from the first :meth:`start` call to
      the most recent :meth:`record` call (useful for end-of-run summaries).

    Usage::

        tracker = ThroughputTracker(window=50, unit="samples")
        tracker.start()
        for batch in dataloader:
            with tracker.step(len(batch)):
                loss = train_step(batch)
            print(tracker.throughput)   # samples/sec (sliding window)
        tracker.log_to_run(step=epoch, prefix="train")

    Args:
        window: Maximum number of recent steps kept for the sliding-window
            throughput calculation.  A larger window gives a smoother but
            less responsive estimate.
        unit: Human-readable label for items (e.g. ``"samples"``,
            ``"steps"``, ``"days"``).  Used in metric keys and ``__repr__``.
    """

    def __init__(self, window: int = 100, unit: str = "items") -> None:
        self.window = window
        self.unit = unit

        # List of (wall_time, n_items) pairs — last `window` entries kept
        self._timestamps: List[Tuple[float, int]] = []
        self._total_items: int = 0
        self._start_time: Optional[float] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> "ThroughputTracker":
        """Record the start time for lifetime throughput calculation."""
        self._start_time = time.perf_counter()
        return self

    def reset(self) -> "ThroughputTracker":
        """Reset all state (timestamps, item count, start time)."""
        self._timestamps.clear()
        self._total_items = 0
        self._start_time = None
        return self

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(self, n_items: int = 1) -> None:
        """Record that *n_items* items have just been processed.

        Args:
            n_items: Number of items completed at this call.
        """
        now = time.perf_counter()
        self._total_items += n_items
        self._timestamps.append((now, n_items))
        if len(self._timestamps) > self.window:
            self._timestamps = self._timestamps[-self.window :]

    @contextlib.contextmanager
    def step(self, n_items: int = 1) -> Generator[None, None, None]:
        """Context manager that times a processing step and records it.

        Records *n_items* on exit (whether or not an exception was raised).

        Args:
            n_items: Number of items processed inside the ``with`` block.

        Example::

            with tracker.step(batch_size):
                loss = model(batch)
        """
        yield
        self.record(n_items)

    # ------------------------------------------------------------------
    # Throughput properties
    # ------------------------------------------------------------------

    @property
    def throughput(self) -> float:
        """Sliding-window throughput (items/sec).

        Computed over the last *window* recorded steps.
        Returns ``0.0`` when fewer than two steps have been recorded.
        """
        if len(self._timestamps) < 2:
            return 0.0
        window_items = sum(n for _, n in self._timestamps)
        window_time = self._timestamps[-1][0] - self._timestamps[0][0]
        if window_time <= 0.0:
            return 0.0
        return window_items / window_time

    @property
    def total_throughput(self) -> float:
        """Average throughput since :meth:`start` was called (items/sec).

        Returns ``0.0`` when :meth:`start` has not been called or no items
        have been recorded.
        """
        if self._start_time is None or self._total_items == 0:
            return 0.0
        elapsed = time.perf_counter() - self._start_time
        if elapsed <= 0.0:
            return 0.0
        return self._total_items / elapsed

    @property
    def total_items(self) -> int:
        """Total number of items recorded since :meth:`start`."""
        return self._total_items

    @property
    def elapsed(self) -> float:
        """Wall-clock seconds elapsed since :meth:`start` (0.0 if not started)."""
        if self._start_time is None:
            return 0.0
        return time.perf_counter() - self._start_time

    # ------------------------------------------------------------------
    # Summary & logging
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Return a dict summarising current throughput figures."""
        return {
            "unit": self.unit,
            "total_items": self._total_items,
            "window_throughput": self.throughput,
            "total_throughput": self.total_throughput,
            "elapsed_sec": self.elapsed,
        }

    def log_to_run(self, step: int = 0, prefix: str = "throughput") -> None:
        """Log sliding-window and total throughput to the active WSTracker run.

        Metrics logged:

        * ``{prefix}/{unit}_per_sec`` — sliding-window throughput
        * ``{prefix}/total_{unit}`` — lifetime item count
        * ``{prefix}/total_throughput`` — lifetime throughput

        Args:
            step: Metric step dimension (e.g. epoch number).
            prefix: Metric key prefix.
        """
        with contextlib.suppress(Exception):
            from tracking.core.run import get_active_run

            run = get_active_run()
            if run is not None:
                run.log_metrics(
                    {
                        f"{prefix}/{self.unit}_per_sec": self.throughput,
                        f"{prefix}/total_{self.unit}": float(self._total_items),
                        f"{prefix}/total_throughput": self.total_throughput,
                    },
                    step=step,
                )

    def __repr__(self) -> str:
        return (
            f"ThroughputTracker(unit={self.unit!r}, "
            f"throughput={self.throughput:.1f} {self.unit}/sec, "
            f"total={self._total_items} {self.unit})"
        )
