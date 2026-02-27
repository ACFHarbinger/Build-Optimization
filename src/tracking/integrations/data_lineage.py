"""Data-lineage tracking callback for simulation runs.

Hooks into the day-by-day simulation loop to record two complementary
streams of information into the centralised WSTracker:

1. **Scalar KPIs per day** — logged via :class:`SimulationRunTracker` so
   every per-day metric (``profit``, ``kg``, ``km``, etc.) is queryable
   as a time-series in the tracking database.

2. **Bin fill distribution snapshots** — logged via
   :class:`RuntimeDataTracker` so the statistical shape of the fill
   levels (mean, std, min, max) is recorded at configurable intervals,
   making distribution drift across the simulation visible in the
   experiment database.

Typical usage
-------------
The callback is created once per ``(policy, sample)`` pair and wired
into :class:`~pipeline.simulations.states.running.RunningState`::

    cb = DataLineageCallback(pol_name, sample_id, log_freq=5)
    cb.on_simulation_start(sim_context)

    for day in range(1, total_days + 1):
        day_ctx = run_day(day_ctx)
        cb.on_step_end(day_ctx, day)
"""

from __future__ import annotations

import contextlib
from typing import Any, Dict, Optional

import torch

from tracking.core.run import get_active_run
from tracking.integrations.data import RuntimeDataTracker
from tracking.integrations.simulation import SimulationRunTracker


class DataLineageCallback:
    """Step-level data lineage and KPI tracker for a simulation run.

    Records two complementary streams per ``(policy, sample_id)`` pair:

    * **Per-day KPIs** (every step) — profit, km, kg, overflows, etc. —
      forwarded to :class:`SimulationRunTracker` as time-series metrics
      so they can be trended and compared across runs.

    * **Bin fill distribution snapshots** (every *log_freq* steps) —
      captured via :class:`RuntimeDataTracker`; records mean/std/min/max
      of fill levels so distribution drift is visible in the DB.

    Args:
        policy_name: Policy name used for metric key namespacing.
        sample_id: Sample/seed index used for metric key namespacing.
        log_freq: Take a fill-distribution snapshot every *log_freq* days
            (default ``1`` = every day).  Raise to ``5`` or ``10`` for
            long simulations to reduce write pressure.
    """

    def __init__(self, policy_name: str, sample_id: int, log_freq: int = 1) -> None:
        self._policy_name = policy_name
        self._sample_id = sample_id
        self._log_freq = max(1, log_freq)
        self._step_count: int = 0
        self._data_tracker: Optional[RuntimeDataTracker] = None
        self._sim_tracker: Optional[SimulationRunTracker] = None

    # ------------------------------------------------------------------
    # Public lifecycle hooks
    # ------------------------------------------------------------------

    def on_simulation_start(self, context: Any) -> None:
        """Snapshot the initial bin fill distribution and attach trackers.

        Should be called **once** before the day loop begins, after the
        simulation context has been fully initialised (bins loaded).

        Args:
            context: :class:`~pipeline.simulations.states.base.context.SimulationContext`
                — ``context.bins`` is accessed to snapshot fill levels.
        """
        run = get_active_run()
        if run is None:
            return

        self._data_tracker = RuntimeDataTracker(run)
        self._sim_tracker = SimulationRunTracker(run, self._policy_name, self._sample_id)

        bins = getattr(context, "bins", None)
        if bins is not None:
            fill_dict = _bins_to_tensor_dict(bins)
            if fill_dict:
                self._data_tracker.on_load(
                    fill_dict,
                    shape=next(iter(fill_dict.values())).shape,
                    metadata={
                        "event": "simulation_start",
                        "policy": self._policy_name,
                        "sample_id": self._sample_id,
                        "variable_name": "bins_fill_distribution",
                        "source_file": "integrations/data_lineage.py",
                        "source_line": 95,
                    },
                )

    def on_step_end(self, day_context: Any, day: int) -> None:
        """Log per-day KPIs and (throttled) fill-distribution snapshots.

        Args:
            day_context: :class:`~pipeline.simulations.day_context.SimulationDayContext`
                — ``daily_log`` dict and ``bins`` are read from here.
            day: Current simulation day index (used as metric step).
        """
        self._step_count += 1

        # --- 1. Log scalar KPIs every day --------------------------------
        if self._sim_tracker is not None:
            dlog: Dict[str, Any] = getattr(day_context, "daily_log", None) or {}
            self._sim_tracker.log_day(day, dlog)

            # Tour length (list → count of stops, not loggable as float directly)
            tour = dlog.get("tour")
            if tour is not None:
                with contextlib.suppress(Exception):
                    self._sim_tracker._run.log_metric(
                        f"{self._sim_tracker._prefix}/tour_len", float(len(tour)), step=day
                    )

            # Cumulative bins scalars not present in dlog (km, profit, kg totals).
            # Logging the cumulative series lets consumers derive per-day deltas.
            bins = getattr(day_context, "bins", None)
            if bins is not None:
                _log_cumulative_bins(self._sim_tracker, bins, day)

        # --- 2. Throttled fill-distribution snapshot ---------------------
        if self._step_count % self._log_freq != 0:
            return

        if self._data_tracker is not None:
            bins = getattr(day_context, "bins", None)
            if bins is not None:
                fill_dict = _bins_to_tensor_dict(bins)
                if fill_dict:
                    self._data_tracker.snapshot(fill_dict, tag=f"sim/day_{day}", step=day)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _log_cumulative_bins(sim_tracker: "SimulationRunTracker", bins: Any, day: int) -> None:
    """Log cumulative bins scalar state as time-series metrics.

    These values are not present in ``daily_log`` because they are accumulated
    on the bins object rather than recorded per-day.  Logging the cumulative
    series at each step lets consumers derive per-day deltas by differencing.

    Scalars logged (all namespaced under ``{policy}/s{sample}/cumul/``):

    * ``km``        — total distance travelled so far (``bins.travel``)
    * ``kg``        — total waste collected (``bins.collected`` sum)
    * ``kg_lost``   — total waste lost to overflow (``bins.lost`` sum)
    * ``overflows`` — total overflow events (``bins.inoverflow`` sum)
    * ``profit``    — cumulative net profit (``bins.profit``)
    """
    prefix = f"{sim_tracker._prefix}/cumul"
    scalars: Dict[str, float] = {}

    with contextlib.suppress(Exception):
        travel = getattr(bins, "travel", None)
        if travel is not None:
            scalars[f"{prefix}/km"] = float(travel)

    with contextlib.suppress(Exception):
        import numpy as np

        collected = getattr(bins, "collected", None)
        if collected is not None:
            scalars[f"{prefix}/kg"] = float(np.sum(collected))

        lost = getattr(bins, "lost", None)
        if lost is not None:
            scalars[f"{prefix}/kg_lost"] = float(np.sum(lost))

        inoverflow = getattr(bins, "inoverflow", None)
        if inoverflow is not None:
            scalars[f"{prefix}/overflows"] = float(np.sum(inoverflow))

    with contextlib.suppress(Exception):
        profit = getattr(bins, "profit", None)
        if profit is not None:
            scalars[f"{prefix}/profit"] = float(profit)

    if scalars:
        sim_tracker._run.log_metrics(scalars, step=day)


def _bins_to_tensor_dict(bins: Any) -> Dict[str, torch.Tensor]:
    """Convert key :class:`~pipeline.simulations.bins.base.Bins`
    numpy arrays to a ``{name: tensor}`` dict for ``RuntimeDataTracker``.

    The four arrays tracked are:

    * ``fill`` — observed fill level per bin (``bins.c``), 0–100 %
    * ``real_fill`` — ground-truth fill level (``bins.real_c``), 0–100 %
    * ``lost`` — cumulative waste lost to overflow per bin
    * ``inoverflow`` — cumulative overflow events per bin
    """
    mapping = {
        "fill": "c",
        "real_fill": "real_c",
        "lost": "lost",
        "inoverflow": "inoverflow",
    }
    result: Dict[str, torch.Tensor] = {}
    for tensor_name, attr in mapping.items():
        with contextlib.suppress(Exception):
            arr = getattr(bins, attr, None)
            if arr is not None:
                result[tensor_name] = torch.as_tensor(arr, dtype=torch.float32)
    return result
