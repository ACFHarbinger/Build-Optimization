"""
TSP Policy module.

Implements a single-vehicle routing policy (TSP) that visits a specific set of bins.
Agnostic to how the targets were selected.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies import TSPConfig
from logic.src.policies.adapters.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.tsp import find_route, get_multi_tour, get_route_cost

from .factory import PolicyRegistry


@PolicyRegistry.register("tsp")
class TSPPolicy(BaseRoutingPolicy):
    """
    Traveling Salesperson Policy (TSP).

    Visits provided 'must_go' bins using a single vehicle strategy with
    capacity-based tour splitting.
    """

    def __init__(self, config: Optional[Union[TSPConfig, Dict[str, Any]]] = None):
        """Initialize TSP policy with optional config.

        Args:
            config: TSPConfig dataclass, raw dict from YAML, or None.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return TSPConfig

    def _get_config_key(self) -> str:
        """Return config key for TSP."""
        return "tsp"

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
        Run TSP solver with capacity-based splitting.
        """
        n_nodes = len(sub_dist_matrix)
        nodes_to_visit = list(range(1, n_nodes))

        # 1. Find a giant tour visiting all potential targets
        # Note: find_route expects global indices usually, but we pass local ones here.
        # It takes C (dist_matrix), to_collect, and time_limit.
        time_limit = values.get("time_limit", 2.0)
        tour = find_route(sub_dist_matrix, nodes_to_visit, time_limit=time_limit)

        # wastes_arr_bins should contain only customer nodes 1..M for get_multi_tour index mapping (x-1)
        wastes_arr_bins = np.array([sub_wastes[i] for i in range(1, n_nodes)])

        # 2. Split the tour greedily based on capacity
        full_tour = get_multi_tour(tour, wastes_arr_bins, capacity, sub_dist_matrix)

        # 3. Convert flat tour to List[List[int]]
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

        # 4. Calculate total distance cost
        total_dist = 0.0
        for r in real_routes:
            full_r = [0] + r + [0]
            total_dist += get_route_cost(sub_dist_matrix, full_r)

        return real_routes, 0.0, total_dist * cost_unit
