"""
CVRP Policy module.

Implements a multi-vehicle routing policy (CVRP) that visits a specific set of bins.
Agnostic to how the targets were selected.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies import CVRPConfig
from logic.src.policies.adapters.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.cvrp import find_routes, find_routes_ortools

from .factory import PolicyRegistry


@PolicyRegistry.register("cvrp")
class CVRPPolicy(BaseRoutingPolicy):
    """
    Capacitated Vehicle Routing Policy (CVRP).

    Visits provided 'must_go' bins using multiple vehicles.
    """

    def __init__(self, config: Optional[Union[CVRPConfig, Dict[str, Any]]] = None):
        """Initialize CVRP policy with optional config.

        Args:
            config: CVRPConfig dataclass, raw dict from YAML, or None.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return CVRPConfig

    def _get_config_key(self) -> str:
        """Return config key for CVRP."""
        return "cvrp"

    def _run_solver(
        self,
        sub_dist_matrix: np.ndarray,
        sub_wastes: Dict[int, float],
        capacity: float,
        revenue: float,
        cost_unit: float,
        values: Dict[str, Any],
        mandatory_nodes: List[int],
        **kwargs: Any,
    ) -> Tuple[List[List[int]], float, float]:
        """
        Run CVRP solver.

        Note: CVRP uses global indices directly, not the subset mapping.
        """
        # Not used - we override execute() for CVRP
        return [[]], 0.0, 0.0

    def execute(self, **kwargs: Any) -> Tuple[List[int], float, Any]:
        """
        Execute CVRP policy.

        Overrides base execute because CVRP uses different solver interface
        and works with global indices directly.
        """
        must_go = kwargs.get("must_go", [])
        early_result = self._validate_must_go(must_go)
        if early_result is not None:
            return early_result

        bins = kwargs["bins"]
        area = kwargs.get("area", "Rio Maior")
        waste_type = kwargs.get("waste_type", "plastic")
        n_vehicles = kwargs.get("n_vehicles", 1)
        cached = kwargs.get("cached")
        coords = kwargs.get("coords")
        config = kwargs.get("config", {})
        distancesC = kwargs.get("distancesC")
        distance_matrix = kwargs.get("distance_matrix", distancesC)

        to_collect = list(must_go) if must_go else list(range(1, bins.n + 1))

        # Load capacity
        capacity, _, _, values = self._load_area_params(area, waste_type, config)
        self._log_solver_params(values, kwargs)

        # Get engine and time_limit from typed config or values dict
        cfg = self._config
        engine = getattr(cfg, "engine", None) if cfg is not None else values.get("engine", "pyvrp")
        if engine is None:
            engine = values.get("engine", "pyvrp")
        time_limit = getattr(cfg, "time_limit", None) if cfg is not None else values.get("time_limit", 2.0)
        if time_limit is None:
            time_limit = values.get("time_limit", 2.0)

        # Use cached route if available and no specific must_go
        if cached is not None and len(cached) > 1 and not must_go:
            tour = cached
        else:
            solver_fn = find_routes_ortools if engine == "ortools" else find_routes
            tour = solver_fn(
                distancesC,
                bins.c,
                capacity,
                np.array(to_collect),
                n_vehicles,
                coords,
                time_limit=time_limit,
            )
        # Ensure list format
        if hasattr(tour, "tolist"):
            tour = tour.tolist()
        elif not isinstance(tour, list):
            tour = list(tour)

        cost = self._compute_cost(distance_matrix, tour)
        return tour, cost, tour
