"""
Policy Visualization Mixin.

Provides lightweight, zero-overhead state recording for classical and
heuristic policies.  Import :class:`PolicyVizMixin` and add it as a
mixin base-class to any policy that should expose iteration data to the
Streamlit dashboard.

Usage::

    class VectorizedALNS(PolicyVizMixin):
        def solve(self, ...):
            for i in range(n_iterations):
                ...
                self._viz_record(iteration=i, best_cost=best_costs.min().item())

    # After running:
    data = solver.get_viz_data()
    # data["iteration"], data["best_cost"]  → lists of recorded values
"""

import contextlib
from typing import Any, Dict, List


class PolicyStateRecorder:
    """
    Fixed-capacity ring-buffer for per-iteration policy telemetry.

    Args:
        max_history: Maximum number of records to keep.  Oldest entries
                     are silently dropped once the buffer is full.
    """

    def __init__(self, max_history: int = 5000) -> None:
        self._buf: Dict[str, List[Any]] = {}
        self._max = max_history

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(self, **kwargs: Any) -> None:
        """Append one snapshot of keyword-named scalar metrics."""
        for key, val in kwargs.items():
            lst = self._buf.setdefault(key, [])
            lst.append(val)
            if len(lst) > self._max:
                lst.pop(0)

    def get(self) -> Dict[str, List[Any]]:
        """Return a shallow copy of the accumulated telemetry."""
        return {k: list(v) for k, v in self._buf.items()}

    def reset(self) -> None:
        """Discard all recorded data."""
        self._buf.clear()

    def __len__(self) -> int:
        if not self._buf:
            return 0
        return max(len(v) for v in self._buf.values())


class PolicyVizMixin:
    """
    Mixin that adds zero-overhead telemetry to policy classes.

    The internal :class:`PolicyStateRecorder` is only created on the
    first call to :meth:`_viz_record`, so policies that never record
    data pay no allocation cost at all.

    Concrete policies should call ``self._viz_record(**metrics)`` inside
    their main iteration loop, then consumers call
    ``policy.get_viz_data()`` to retrieve the captured time-series.
    """

    # ------------------------------------------------------------------
    # Lazy recorder property
    # ------------------------------------------------------------------

    @property
    def _viz(self) -> PolicyStateRecorder:
        try:
            return self.__viz_recorder  # type: ignore[attr-defined]
        except AttributeError:
            self.__viz_recorder: PolicyStateRecorder = PolicyStateRecorder()
            return self.__viz_recorder

    # ------------------------------------------------------------------
    # Public API (forwarded to recorder)
    # ------------------------------------------------------------------

    def _viz_record(self, **kwargs: Any) -> None:
        """Record one snapshot of metrics.  Call once per iteration."""
        self._viz.record(**kwargs)

    def get_viz_data(self) -> Dict[str, List[Any]]:
        """
        Return accumulated telemetry as a plain dict of lists.

        Returns an empty dict if ``_viz_record`` was never called.
        """
        try:
            return self.__viz_recorder.get()  # type: ignore[attr-defined]
        except AttributeError:
            return {}

    def reset_viz(self) -> None:
        """Clear all recorded telemetry (safe to call before re-use)."""
        with contextlib.suppress(AttributeError):
            self.__viz_recorder.reset()  # type: ignore[attr-defined]
