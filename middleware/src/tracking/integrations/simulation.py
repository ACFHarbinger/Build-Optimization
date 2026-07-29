"""Simulation pipeline hooks for WSTracker.

Provides :class:`SimulationRunTracker`, a lightweight helper that wraps an
active :class:`~tracking.core.run.Run` and adds simulation-specific
logging conventions (per-day metrics, per-policy aggregates, final results).
"""

from __future__ import annotations

import contextlib
import time
from typing import Any, Dict, List, Optional

from tracking.core.run import Run, get_active_run


class SimulationRunTracker:
    """Simulation-specific metric logger wrapping an active :class:`Run`.

    Each instance corresponds to one ``(policy, sample_id)`` pair inside
    a larger simulation experiment.  All metric keys are namespaced as::

        {policy_name}/sample_{sample_id}/{metric}

    so multiple policies and samples can coexist inside a single parent run.

    Args:
        run: The active tracking run.
        policy_name: Name of the routing policy being evaluated.
        sample_id: Index of the simulation instance/seed.
    """

    def __init__(self, run: Run, policy_name: str, sample_id: int) -> None:
        self._run = run
        self.policy_name = policy_name
        self.sample_id = sample_id
        self._start_wall = time.monotonic()
        self._prefix = f"{policy_name}/s{sample_id}"

    # ------------------------------------------------------------------
    # Per-day logging
    # ------------------------------------------------------------------

    def log_day(self, day: int, metrics: Dict[str, Any]) -> None:
        """Log scalar metrics for a single simulation day.

        Args:
            day: The day index (used as the ``step`` dimension).
            metrics: Key/value pairs of day metrics
                (non-numeric values are silently ignored).
        """
        tagged: Dict[str, float] = {}
        for k, v in metrics.items():
            with contextlib.suppress(TypeError, ValueError):
                tagged[f"{self._prefix}/{k}"] = float(v)
        if tagged:
            self._run.log_metrics(tagged, step=day)

    # ------------------------------------------------------------------
    # Final aggregated result
    # ------------------------------------------------------------------

    def log_final(self, metric_keys: List[str], metric_values: List[Any]) -> None:
        """Log the end-of-simulation aggregated metrics.

        Args:
            metric_keys: Ordered list of metric names (e.g. ``SIM_METRICS``).
            metric_values: Corresponding numeric values.
        """
        prefix = f"{self._prefix}/final"
        for key, val in zip(metric_keys, metric_values):
            with contextlib.suppress(TypeError, ValueError):
                self._run.log_metric(f"{prefix}/{key}", float(val), step=self.sample_id)

        elapsed = time.monotonic() - self._start_wall
        self._run.log_metric(f"{prefix}/wall_time_s", elapsed, step=self.sample_id)
        self._run.flush()

    # ------------------------------------------------------------------
    # Error recording
    # ------------------------------------------------------------------

    def log_failure(self, error: str) -> None:
        """Tag the parent run with a failure notice for this (policy, sample)."""
        tag_key = f"error.{self.policy_name}.s{self.sample_id}"
        self._run.set_tag(tag_key, error[:250])


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def get_sim_tracker(
    policy_name: str,
    sample_id: int,
) -> Optional[SimulationRunTracker]:
    """Return a :class:`SimulationRunTracker` for the active run, or ``None``.

    This is the primary entry-point used inside simulation worker processes
    to obtain a properly namespaced tracker without access to the parent
    Tracker instance.

    Args:
        policy_name: Policy name string.
        sample_id: Sample/seed index.
    """
    run = get_active_run()
    if run is None:
        return None
    return SimulationRunTracker(run, policy_name, sample_id)
