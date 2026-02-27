"""
LKH Policy Adapter.

Uses Lin-Kernighan-Helsgaun heuristic for TSP optimization.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies import LKHConfig
from logic.src.policies.adapters.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.operators.heuristics.lin_kernighan_helsgaun import solve_lkh
from logic.src.policies.tsp import get_multi_tour, get_route_cost

from .factory import PolicyRegistry


@PolicyRegistry.register("lkh")
class LKHPolicy(BaseRoutingPolicy):
    """
    Lin-Kernighan-Helsgaun heuristic policy class.

    Uses LKH-tour improvement for TSP with capacity-based splitting.
    """

    def __init__(self, config: Optional[Union[LKHConfig, Dict[str, Any]]] = None):
        """Initialize LKH policy with optional config.

        Args:
            config: LKHConfig dataclass, raw dict from YAML, or None.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return LKHConfig

    def _get_config_key(self) -> str:
        """Return config key for LKH."""
        return "lkh"

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
        Run LKH solver.

        Returns:
            Tuple of (routes, profit, solver_cost)
        """
        # Convert sub_wastes dict to array for solve_lkh
        n_nodes = len(sub_dist_matrix)
        wastes_arr = np.zeros(n_nodes)
        for i, d in sub_wastes.items():
            wastes_arr[i] = d

        # solve_lkh returns a single tour (closed, [0, ..., 0])
        tour, _ = solve_lkh(
            sub_dist_matrix,
            waste=wastes_arr,
            capacity=capacity,
            max_iterations=values.get("max_iterations", 100),
        )

        # wastes_arr_bins should contain only customer nodes 1..M for get_multi_tour index mapping (x-1)
        wastes_arr_bins = np.array([sub_wastes[i] for i in range(1, n_nodes)])

        # Split into multiple routes if capacity is violated
        full_tour = get_multi_tour(tour, wastes_arr_bins, capacity, sub_dist_matrix)

        # Convert flat tour [0, 1, 2, 0, 3, 0] to List[List[int]]
        real_routes: List[List[int]] = []
        curr_route: List[int] = []
        for node in full_tour:
            if node == 0:
                if curr_route:
                    real_routes.append(curr_route)
                    curr_route = []
            else:
                curr_route.append(node)
        if curr_route:
            real_routes.append(curr_route)

        # Calculate total distance cost
        total_dist = 0.0
        for r in real_routes:
            full_r = [0] + r + [0]
            total_dist += get_route_cost(sub_dist_matrix, full_r)

        return real_routes, 0.0, total_dist * cost_unit
