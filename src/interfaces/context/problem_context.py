"""
ProblemContext — Typed problem-instance descriptor.

Carries all static and dynamic problem data for a single day's execution.
Replaces the untyped kwargs-passing pattern in BaseRoutingPolicy.execute.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from logic.src.pipeline.simulations.bins.prediction import ScenarioTree


@dataclass(frozen=True)
class ProblemContext:
    """
    Complete problem descriptor for one day of the MPVRPP simulation.

    Attributes:
        distance_matrix:   Full (N+1)×(N+1) distance matrix; index 0 = depot.
        wastes:            {bin_id (1-based) → current fill level}.
        fill_rate_means:   shape (V,) — per-bin expected daily increment E[δ_{i,d}].
        fill_rate_stds:    shape (V,) — per-bin std dev of daily increment.
        capacity:          Vehicle capacity Q.
        max_fill:          Maximum bin fill level τ.
        revenue_per_kg:    r_w — revenue coefficient.
        cost_per_km:       c_km — distance cost coefficient.
        horizon:           D — planning horizon (days).
        mandatory:         Bin IDs (1-based) that MUST be visited today.
        locations:         shape (N+1, 2) — [lng, lat] per node; index 0 = depot.
        scenario_tree:     Stochastic fill-level scenario tree (may be None for
                           single-day or deterministic solvers).
        area:              Geographic area name (used for parameter loading).
        waste_type:        Waste type string (used for parameter loading).
        n_vehicles:        Number of available vehicles (None = unlimited).
        seed:              Random seed for reproducibility.
        day_index:         Current day index in the simulation (0-based).
        extra:             Catch-all for solver-specific or area-specific fields
                           that do not fit the standard attributes above.
    """

    distance_matrix: np.ndarray
    wastes: Dict[int, float]
    fill_rate_means: np.ndarray
    fill_rate_stds: np.ndarray
    capacity: float
    max_fill: float
    revenue_per_kg: float
    cost_per_km: float
    horizon: int
    mandatory: List[int]
    locations: np.ndarray
    scenario_tree: Optional[ScenarioTree] = None
    area: str = "Rio Maior"
    waste_type: str = "plastic"
    n_vehicles: Optional[int] = None
    seed: int = 42
    day_index: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def current_day(self) -> int:
        """Alias for day_index (legacy compatibility)."""
        return self.day_index

    @property
    def fill_rates(self) -> np.ndarray:
        """Alias for fill_rate_means (legacy compatibility)."""
        return self.fill_rate_means

    @classmethod
    def from_kwargs(
        cls, kwargs: Dict[str, Any], capacity: float, revenue_per_kg: float, cost_per_km: float
    ) -> "ProblemContext":
        """
        Construct a ProblemContext from the raw kwargs dict passed into execute().

        `capacity`, `revenue_per_kg`, and `cost_per_km` are passed explicitly
        because they come from `_load_area_params`, not directly from kwargs.

        Args:
            kwargs:          The raw kwargs dict from execute().
            capacity:        Q loaded from area params.
            revenue_per_kg:  r_w loaded from area params.
            cost_per_km:     c_km loaded from area params.
        """
        bins = kwargs.get("bins")
        wastes: Dict[int, float] = kwargs.get("wastes") or (
            {i: float(bins.c[i - 1]) for i in range(1, len(bins.c) + 1)} if bins is not None else {}
        )

        coords = kwargs.get("coords")
        n = len(kwargs["distance_matrix"]) - 1
        locations = np.zeros((n + 1, 2))
        if coords is not None:
            try:
                ci = coords.set_index("ID")
                depot_row = coords[coords["ID"] == 0]
                if len(depot_row) > 0:
                    locations[0] = [float(depot_row["Lng"].iloc[0]), float(depot_row["Lat"].iloc[0])]
                else:
                    locations[0] = [float(coords["Lng"].mean()), float(coords["Lat"].mean())]
                for local_idx in range(1, n + 1):
                    global_id = local_idx
                    if global_id in ci.index:
                        locations[local_idx] = [float(ci.loc[global_id, "Lng"]), float(ci.loc[global_id, "Lat"])]
            except Exception:
                pass  # locations stays zeros; algorithms degrade gracefully

        # Fill-rate statistics (may be absent for single-day simulations)
        fill_rate_means = np.zeros(n)
        fill_rate_stds = np.zeros(n)
        if bins is not None:
            if hasattr(bins, "rate") and bins.rate is not None:
                fill_rate_means = np.asarray(bins.rate, dtype=float)
            if hasattr(bins, "rate_std") and bins.rate_std is not None:
                fill_rate_stds = np.asarray(bins.rate_std, dtype=float)

        n_vehicles_raw = kwargs.get("n_vehicles")
        n_vehicles = int(n_vehicles_raw) if n_vehicles_raw is not None and int(n_vehicles_raw) > 0 else None

        return cls(
            distance_matrix=kwargs["distance_matrix"],
            wastes=wastes,
            fill_rate_means=fill_rate_means,
            fill_rate_stds=fill_rate_stds,
            capacity=capacity,
            max_fill=float(getattr(bins, "max_fill", 100.0)) if bins is not None else 100.0,
            revenue_per_kg=revenue_per_kg,
            cost_per_km=cost_per_km,
            horizon=kwargs.get("horizon", 7),
            mandatory=kwargs.get("mandatory", []),
            locations=locations,
            scenario_tree=kwargs.get("scenario_tree"),
            area=kwargs.get("area", "Rio Maior"),
            waste_type=kwargs.get("waste_type", "plastic"),
            n_vehicles=n_vehicles,
            seed=kwargs.get("seed", 42),
            day_index=kwargs.get("day_index", 0),
            extra={
                k: v
                for k, v in kwargs.items()
                if k
                not in {
                    "distance_matrix",
                    "wastes",
                    "bins",
                    "coords",
                    "scenario_tree",
                    "area",
                    "waste_type",
                    "n_vehicles",
                    "seed",
                    "day_index",
                    "mandatory",
                    "horizon",
                    "search_context",
                    "multi_day_context",
                    "config",
                }
            },
        )

    def advance(self, route: List[int], delta: Optional[np.ndarray] = None) -> "ProblemContext":
        """
        Return a new ProblemContext for day d+1 after executing `route` on day d.

        Args:
            route:  Ordered list of bin IDs visited (1-based, depot excluded).
            delta:  Optional shape (V,) array of actual fill increments for day d+1.
                    If None, `fill_rate_means` is used as a deterministic surrogate.
        """
        visited = set(route)
        increments = delta if delta is not None else self.fill_rate_means
        new_wastes: Dict[int, float] = {}
        for i, w in self.wastes.items():
            idx = i - 1
            inc = float(increments[idx]) if idx < len(increments) else 0.0
            if i in visited:
                new_wastes[i] = min(inc, self.max_fill)
            else:
                new_wastes[i] = min(w + inc, self.max_fill)
        return ProblemContext(
            distance_matrix=self.distance_matrix,
            wastes=new_wastes,
            fill_rate_means=self.fill_rate_means,
            fill_rate_stds=self.fill_rate_stds,
            capacity=self.capacity,
            max_fill=self.max_fill,
            revenue_per_kg=self.revenue_per_kg,
            cost_per_km=self.cost_per_km,
            horizon=self.horizon,
            mandatory=[],  # mandatory is day-specific; caller sets it
            locations=self.locations,
            scenario_tree=self.scenario_tree,
            area=self.area,
            waste_type=self.waste_type,
            n_vehicles=self.n_vehicles,
            seed=self.seed,
            day_index=self.day_index + 1,
            extra=self.extra,
        )
