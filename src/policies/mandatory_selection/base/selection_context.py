"""
Selection Context Module.

This module defines the `SelectionContext` data class, which serves as a
container for all relevant data needed by selection strategies to make decisions.

Attributes:
    None

Example:
    >>> from logic.src.policies.mandatory.base.selection_context import SelectionContext
    >>> ctx = SelectionContext(bin_ids=np.array([1, 2]), current_fill=np.array([0.8, 0.5]))
"""

from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np
from numpy.typing import NDArray


@dataclass
class SelectionContext:
    """
    Context container for all potential inputs required by selection strategies.
    """

    bin_ids: NDArray[np.int32]
    current_fill: NDArray[np.float64]
    accumulation_rates: Optional[NDArray[np.float64]] = None
    std_deviations: Optional[NDArray[np.float64]] = None
    current_day: int = 0
    threshold: float = 0.0
    next_collection_day: Optional[int] = None
    distance_matrix: Optional[NDArray[Any]] = None
    paths_between_states: Optional[List[List[List[int]]]] = None
    vehicle_capacity: float = 0.0
    revenue_kg: float = 0.0
    bin_density: float = 0.0
    bin_volume: float = 0.0
    max_fill: float = 100.0
    overflow_penalty_frac: float = 1.0  # default to 100% of bin waste capacity

    # --- New fields for advanced strategies ---
    # DeadlineDrivenSelection / MultiDayOverflowSelection
    horizon_days: int = 3
    # SpatialSynergySelection
    critical_threshold: float = 0.90
    synergy_threshold: float = 0.60
    radius: float = 10.0

    # --- Knapsack / economic coupling ---
    n_vehicles: int = 1
    cost_per_km: float = 0.0
    use_eoq_threshold: bool = False
    holding_cost_per_kg_day: float = 0.0
    ordering_cost_per_visit: float = 0.0

    # --- Rollout strategy ---
    rollout_horizon: int = 5
    rollout_base_policy: str = "last_minute"
    rollout_n_scenarios: int = 1  # 1 = deterministic rollout; >1 = Monte Carlo
    rollout_discount: float = 0.95

    # --- Whittle index ---
    whittle_discount: float = 0.95
    whittle_grid_size: int = 21  # number of subsidy values probed for index computation

    # --- CVaR ---
    cvar_alpha: float = 0.95  # tail level; strategy compares CVaR to context.threshold

    # --- Savings / Clarke-Wright ---
    savings_min_fill_ratio: float = 0.5

    # --- Set-cover ---
    service_radius: float = 5.0

    # --- (Super/Sub-)Modular greedy ---
    modular_alpha: float = 1.0  # trade-off between revenue and route length
    modular_budget: int = 0  # 0 = unbounded; otherwise cardinality cap

    # --- Learned / imitation selection ---
    learned_model_path: Optional[str] = None
    learned_threshold: float = 0.5  # classification cutoff

    # --- Contextual Thompson sampling dispatcher ---
    dispatcher_state_path: Optional[str] = None  # pickle of posterior params
    dispatcher_candidate_strategies: Optional[List[str]] = None
    dispatcher_exploration: float = 1.0  # posterior-sample temperature
    dispatcher_mode: str = "union"

    # --- Distributionally robust ---
    wasserstein_radius: float = 0.1
    wasserstein_p: int = 1  # only p=1 is implemented

    # --- Stochastic scenario tree ---
    # Optional ScenarioTree (pipeline.simulations.bins.prediction.ScenarioTree) injected
    # by the simulation engine before calling select_bins().  Strategies that do not
    # consume it simply ignore the field.
    scenario_tree: Optional[Any] = None
