"""
Joint Selection–Construction Context Module.

This module defines :class:`JointSelectionConstructionContext`, a unified
data-container that carries all inputs required by algorithms that solve
mandatory-bin selection **and** route construction simultaneously within
a single optimisation loop (e.g., NDS-BRKGA).

It mirrors the scalar fields of :class:`SelectionContext` (for the bin-level
packing sub-problem) and augments them with routing fields (distance matrix,
revenue, cost) so that the joint solver can evaluate both objectives without
switching between two separate context objects.

Example::

    >>> from logic.src.policies.selection_and_construction.base.joint_context import (
    ...     JointSelectionConstructionContext,
    ... )
    >>> ctx = JointSelectionConstructionContext(
    ...     bin_ids=np.array([1, 2, 3]),
    ...     current_fill=np.array([80.0, 45.0, 95.0]),
    ...     distance_matrix=dm,
    ...     capacity=500.0,
    ...     revenue_kg=0.12,
    ...     cost_per_km=0.05,
    ... )
"""

from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np
from numpy.typing import NDArray


@dataclass
class JointSelectionConstructionContext:
    """
    Unified context for joint bin-selection and route-construction solvers.

    Attributes:
        bin_ids: 1-based global bin IDs for all candidate bins.  Shape ``(N,)``.
        current_fill: Current fill levels as percentages ``[0, 100]``.
            Shape ``(N,)``.
        distance_matrix: Full ``(N+1) × (N+1)`` distance matrix (index 0 = depot).
        capacity: Single-vehicle carrying capacity in kg.
        revenue_kg: Revenue per kg of collected waste (€/kg).
        cost_per_km: Travel cost per distance unit (€/km or €/unit).

        accumulation_rates: Optional per-bin daily fill-rate predictions.
            Shape ``(N,)``.
        std_deviations: Optional per-bin fill-rate standard deviations.
            Shape ``(N,)``.
        bin_density: Waste density in kg/L (used to convert fill-% to kg).
        bin_volume: Bin volume in litres.
        max_fill: Maximum fill level (default 100.0 = full).
        overflow_penalty_frac: Penalty for overflowing bins expressed as a
            fraction of the bin's full waste capacity.  Default ``1.0``.

        n_vehicles: Number of available vehicles (default 1).
        current_day: Simulation day index (0-based).
        horizon_days: Lookahead horizon for overflow prediction.
        scenario_tree: Optional ``ScenarioTree`` from the prediction engine.
            Strategies that do not consume it may ignore this field.

        paths_between_states: Optional pre-computed shortest paths.
        mandatory_override: Optional list of 1-based bin IDs that MUST be
            collected regardless of the solver's selection decision.
    """

    # ------------------------------------------------------------------
    # Core bin-level inputs
    # ------------------------------------------------------------------
    bin_ids: NDArray[np.int32]
    current_fill: NDArray[np.float64]
    distance_matrix: NDArray[Any]
    capacity: float
    revenue_kg: float
    cost_per_km: float

    # ------------------------------------------------------------------
    # Optional enrichment (mirrors SelectionContext)
    # ------------------------------------------------------------------
    accumulation_rates: Optional[NDArray[np.float64]] = None
    std_deviations: Optional[NDArray[np.float64]] = None
    bin_density: float = 0.0
    bin_volume: float = 0.0
    max_fill: float = 100.0
    overflow_penalty_frac: float = 1.0

    # ------------------------------------------------------------------
    # Fleet / scheduling
    # ------------------------------------------------------------------
    n_vehicles: int = 1
    current_day: int = 0
    horizon_days: int = 3

    # ------------------------------------------------------------------
    # Stochastic enrichment
    # ------------------------------------------------------------------
    scenario_tree: Optional[Any] = None

    # ------------------------------------------------------------------
    # Routing extras
    # ------------------------------------------------------------------
    paths_between_states: Optional[List[List[List[int]]]] = None
    mandatory_override: Optional[List[int]] = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def bin_mass_kg(self) -> NDArray[np.float64]:
        """Full waste mass per bin in kg: ``density × volume``."""
        return np.full(len(self.bin_ids), self.bin_density * self.bin_volume, dtype=float)

    def revenue_scaled(self) -> float:
        """
        Revenue per 1 % fill unit.

        Converts ``revenue_kg`` (€/kg) to € per percentage point of fill
        so it can be directly multiplied by fill-level differences.

        Returns:
            float: ``revenue_kg × density × volume / 100``.
        """
        return self.revenue_kg * (self.bin_density * self.bin_volume / 100.0)
