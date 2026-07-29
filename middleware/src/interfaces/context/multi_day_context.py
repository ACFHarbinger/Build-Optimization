"""
MultiDayContext — Rolling Horizon State Tracking Ledger.

This module defines the MultiDayContext which allows routing policies to access
historical data and metadata from previous days in a rolling horizon execution.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class MultiDayContext:
    """
    State ledger for rolling horizon multi-day optimization.

    Attributes:
        day_index: current day index in the simulation.
        previous_days_metadata: List of metadata dictionaries from previous days.
        full_plan_snapshot: Optional snapshot of the T-period plan from previous days.
        accumulated_stats: dict for tracking cross-day metrics (e.g. cumulative profit).
        extra: catch-all for custom policy-specific multi-day state.
    """

    day_index: int = 0
    previous_days_metadata: List[Dict[str, Any]] = field(default_factory=list)
    full_plan_snapshot: Optional[List[List[List[int]]]] = None  # [day][tour][node]
    accumulated_stats: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def initialize(cls, day_index: int = 0) -> "MultiDayContext":
        """Initialize a fresh multi-day context."""
        return cls(day_index=day_index)

    def update(self, **patch: Any) -> "MultiDayContext":
        """Return a new MultiDayContext with updated fields."""
        return MultiDayContext(
            day_index=patch.get("day_index", self.day_index),
            previous_days_metadata=patch.get("previous_days_metadata", self.previous_days_metadata),
            full_plan_snapshot=patch.get("full_plan_snapshot", self.full_plan_snapshot),
            accumulated_stats=patch.get("accumulated_stats", self.accumulated_stats),
            extra=patch.get("extra", self.extra),
        )
