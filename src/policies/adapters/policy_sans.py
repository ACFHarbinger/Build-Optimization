"""
SANS Policy Adapter (Simulated Annealing Neighborhood Search).

Uses Simulated Annealing for route optimization.
Supports two engines:
  - 'new': Improved SA with initial solution and iterative refinement
  - 'og': Original look-ahead algorithm for collection (LAC)
"""

from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd

from logic.src.configs.policies import SANSConfig
from logic.src.constants.routing import (
    DEFAULT_COMBINATION,
    DEFAULT_SHIFT_DURATION,
    DEFAULT_TIME_LIMIT,
    DEFAULT_V_VALUE,
)
from logic.src.data.processor import convert_to_dict
from logic.src.policies.adapters.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.simulated_annealing_neighborhood_search import (
    improved_simulated_annealing,
)
from logic.src.policies.simulated_annealing_neighborhood_search.common.routes import create_points
from logic.src.policies.simulated_annealing_neighborhood_search.common.solution_initialization import (
    compute_initial_solution,
)
from logic.src.policies.simulated_annealing_neighborhood_search.refinement.route_search import find_solutions
from logic.src.policies.tsp import get_route_cost

from .factory import PolicyRegistry


@PolicyRegistry.register("sans")
@PolicyRegistry.register("lac")  # Backward compatibility alias
class SANSPolicy(BaseRoutingPolicy):
    """
    Simulated Annealing Neighborhood Search policy class.

    Uses SA optimization with custom initialization and must-go enforcement.
    Supports two engines via the 'engine' parameter:
      - 'new': Improved simulated annealing with initial solution computation
      - 'og': Original look-ahead collection (LAC) algorithm
    """

    def __init__(self, config: Optional[Union[SANSConfig, Dict[str, Any]]] = None):
        """Initialize SANS policy with optional config.

        Args:
            config: SANSConfig dataclass, raw dict from YAML, or None.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return SANSConfig

    def _get_config_key(self) -> str:
        """Return config key for SANS."""
        return "sans"

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
        """Not used - SANS requires specialized execute()."""
        return [[]], 0.0, 0.0

    def execute(self, **kwargs: Any) -> Tuple[List[int], float, Any]:
        """
        Execute the SANS policy.

        Uses specialized data preparation for simulated annealing.
        """
        # Determine engine from typed config, raw config, or kwargs
        cfg = self._config
        if cfg is not None:
            engine: Literal["new", "og"] = cfg.engine
        else:
            config = kwargs.get("config", {})
            sans_config = config.get("sans", config.get("lac", {}))
            raw_engine = kwargs.get("engine", sans_config.get("engine", "new"))
            engine = "og" if raw_engine == "og" else "new"

        if engine == "og":
            return self._execute_og(**kwargs)
        else:
            return self._execute_new(**kwargs)

    def _execute_new(self, **kwargs: Any) -> Tuple[List[int], float, Any]:
        """Execute the improved simulated annealing engine."""
        must_go = kwargs.get("must_go", [])
        early_result = self._validate_must_go(must_go)
        if early_result is not None:
            return early_result

        bins = kwargs["bins"]
        distance_matrix = kwargs["distance_matrix"]
        coords = kwargs["coords"]
        area = kwargs.get("area", "Rio Maior")
        waste_type = kwargs.get("waste_type", "plastic")
        config = kwargs.get("config", {})

        # Load area parameters
        Q, R, C, values = self._load_area_params(area, waste_type, config)
        self._log_solver_params(values, kwargs)

        B = values.get("B", 0)
        V = values.get("V", 0)

        # Prepare data for SANS logic
        data = pd.DataFrame(
            {
                "ID": np.arange(1, bins.n + 1),
                "#bin": np.arange(1, bins.n + 1),
                "Stock": bins.c.astype("float32"),
                "Accum_Rate": bins.means.astype("float32"),
            }
        )
        # Add depot
        depot_row = pd.DataFrame([{"#bin": 0, "Stock": 0.0, "Accum_Rate": 0.0}])
        data = pd.concat([depot_row, data], ignore_index=True)

        coords_dict = convert_to_dict(coords)
        id_to_index = {i: i for i in range(len(distance_matrix))}

        initial_routes = compute_initial_solution(data, coords_dict, distance_matrix, Q, id_to_index)
        current_route = initial_routes[0] if initial_routes else [0, 0]

        # Ensure must_go bins are in the route
        must_go_1 = list(must_go)
        route_set = set(current_route)
        for b in must_go_1:
            if b not in route_set:
                current_route.insert(1, b)

        # Get SA parameters from typed config or raw config
        cfg = self._config
        if cfg is not None:
            sa_params = (cfg.T_init, cfg.iterations_per_T, cfg.alpha, cfg.T_min)
            time_limit = cfg.time_limit
            perc_overflow = cfg.perc_bins_can_overflow
        else:
            sans_config = config.get("sans", {})
            sa_params = (
                sans_config.get("T_init", 75),
                sans_config.get("iterations_per_T", 5000),
                sans_config.get("alpha", 0.95),
                sans_config.get("T_min", 0.01),
            )
            time_limit = values.get("time_limit", 60)
            perc_overflow = sans_config.get("perc_bins_can_overflow", 0.0)

        optimized_routes, best_profit, last_distance, _, _ = improved_simulated_annealing(
            [current_route],
            distance_matrix,
            time_limit,
            id_to_index,
            data,
            Q,
            *sa_params,
            R,
            V,
            B,
            C,
            must_go_1,
            perc_bins_can_overflow=perc_overflow,
            volume=V,
            density_val=B,
            max_vehicles=1,
        )

        tour = optimized_routes[0] if optimized_routes else [0, 0]
        return tour, last_distance, {"profit": best_profit}

    def _execute_og(self, **kwargs: Any) -> Tuple[List[int], float, Any]:
        """Execute the original LAC (look-ahead collection) engine."""
        must_go = kwargs.get("must_go", [])
        early_result = self._validate_must_go(must_go)
        if early_result is not None:
            return early_result

        bins = kwargs["bins"]
        distance_matrix = kwargs["distance_matrix"]
        coords = kwargs["coords"]
        new_data = kwargs.get("new_data")  # Expecting the dataframe
        area = kwargs.get("area", "Rio Maior")
        waste_type = kwargs.get("waste_type", "plastic")
        config = kwargs.get("config", {})
        lac_config = config.get("lac", config.get("sans", {}))

        # Load area parameters
        Q, R, C, area_values = self._load_area_params(area, waste_type, config)
        self._log_solver_params(area_values, kwargs)
        V = lac_config.get("V", DEFAULT_V_VALUE)

        values = {
            "Q": Q,
            "R": R,
            "B": area_values.get("B", 0),
            "C": C,
            "V": V,
            "shift_duration": DEFAULT_SHIFT_DURATION,
            "perc_bins_can_overflow": 0,
        }
        values.update(lac_config)

        # must_go bins are 0-based, find_solutions expects 1-based
        must_go_1 = [b + 1 for b in must_go]

        # Handle case where new_data might not be passed
        if new_data is None:
            # Build minimal dataframe similar to _execute_new
            new_data = pd.DataFrame(
                {
                    "ID": np.arange(1, bins.n + 1),
                    "#bin": np.arange(1, bins.n + 1),
                    "Stock": bins.c.astype("float32"),
                    "Accum_Rate": bins.means.astype("float32"),
                }
            )
            depot_row = pd.DataFrame([{"#bin": 0, "Stock": 0.0, "Accum_Rate": 0.0}])
            new_data = pd.concat([depot_row, new_data], ignore_index=True)

        points = create_points(new_data, coords)

        # Get combination and time_limit from typed config or raw config
        cfg = self._config
        combination = (
            cfg.combination
            if cfg is not None and cfg.combination
            else lac_config.get("combination", DEFAULT_COMBINATION)
        )
        og_time_limit = cfg.time_limit if cfg is not None else lac_config.get("time_limit", DEFAULT_TIME_LIMIT)

        try:
            res, _, _ = find_solutions(
                new_data,
                coords,
                distance_matrix,
                combination,
                must_go_1,
                values,
                bins.n,
                points,
                time_limit=og_time_limit,
            )
        except Exception:
            return [0, 0], 0.0, None

        tour = res[0] if res else [0, 0]
        cost = get_route_cost(distance_matrix, tour)

        # Compute profit: collected revenue - distance cost
        visited = {n for n in tour if n != 0}
        collected_revenue = sum(float(bins.c[n - 1]) * R for n in visited if 1 <= n <= bins.n)
        profit = collected_revenue - cost * C

        return tour, cost, {"profit": profit}
