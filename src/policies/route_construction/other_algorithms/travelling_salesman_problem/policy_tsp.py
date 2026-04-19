"""
TSP Policy module.

Implements a single-vehicle routing policy (TSP) that visits a specific set of bins.
Agnostic to how the targets were selected.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies import TSPConfig
from logic.src.enums import GlobalRegistry, PolicyTag
from logic.src.policies.route_construction.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.route_construction.base.factory import RouteConstructorRegistry

from .params import TSPParams
from .tsp import find_route, get_multi_tour, get_route_cost


@GlobalRegistry.register(
    PolicyTag.HEURISTIC,
    PolicyTag.TRAJECTORY_BASED,
    PolicyTag.LOCAL_SEARCH,
    PolicyTag.CONSTRUCTION,
    PolicyTag.PROFIT_AWARE,
)
@RouteConstructorRegistry.register("tsp")
class TSPPolicy(BaseRoutingPolicy):
    """
    Traveling Salesperson Policy (TSP) with Greedy Tour Splitting.

    This policy implements a single-vehicle routing strategy designed for
    instances where a long, spatial-efficient 'giant tour' can be split into
    multiple feasible vehicle routes.

    Algorithm Logic:
    1.  **Giant Tour Construction**: Solves a Traveling Salesperson Problem (TSP)
        over all mandatory bins to find a continuous path that visits every target
        exactly once, minimizing total travel distance.
    2.  **Greedy Splitting**: Sequentially traverses the giant tour and partitions
        it into separate vehicle routes whenever the cumulative bin load exceeds
        the vehicle capacity. Each partition returns to the depot before
        starting the next segment of the tour.

    The policy relies on efficient underlying TSP solvers (e.g., OR-Tools or
    Christofides) and ensures robust collection even with limited vehicle
    capacities.

    Registry key: ``"tsp"``
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
        Execute the Travelling Salesman Problem (TSP) solver logic with tour splitting.

        This policy solves the TSP over a subset of nodes using a two-stage approach:
        1. Giant Tour Construction: Finds a near-optimal tour visiting all
           mandatory nodes using an efficient TSP heuristic (e.g., Lin-Kernighan
           inspired or OR-Tools).
        2. Capacity-Based Splitting: Partitions the giant tour into feasible
           vehicle routes based on the current collection capacity. This ensures
           that the final plan respects the physical constraints of the waste
           collection fleet while maintaining the spatial efficiency of the
           giant tour.

        Args:
            sub_dist_matrix (np.ndarray): Symmetric distance matrix for the current
                sub-problem nodes.
            sub_wastes (Dict[int, float]): Mapping of local node indices to their
                current bin inventory levels.
            capacity (float): Maximum vehicle collection capacity.
            revenue (float): Revenue obtained per kilogram of waste collected.
            cost_unit (float): Monetary cost incurred per kilometer traveled.
            values (Dict[str, Any]): Merged configuration dictionary containing
                TSP parameters (time_limit, engine).
            mandatory_nodes (List[int]): Local indices of bins that MUST be
                collected in this period.
            **kwargs: Additional context, including:
                - search_context (Optional[SearchContext]): Context for tracking
                  recursive solver statistics.
                - multi_day_context (Optional[MultiDayContext]): Context for
                  inter-day state propagation.

        Returns:
            Tuple[List[List[int]], float, float]: A 3-tuple containing:
                - routes: Optimized collection routes (list-of-lists, local indices).
                - profit: Total calculated net profit (Total Revenue - Total Cost).
                - cost: Total travel cost calculated by the solver.
        """
        # 1. Initialize type-safe Params
        params = TSPParams.from_config(self._config or values)

        # 2. Find a giant tour visiting all potential targets
        # Note: find_route expects global indices usually, but we pass local ones here.
        # It takes C (dist_matrix), to_collect, and time_limit.
        nodes_to_visit = mandatory_nodes
        tour = find_route(sub_dist_matrix, nodes_to_visit, time_limit=params.time_limit)

        # wastes_arr_bins should contain only customer nodes 1..M for get_multi_tour index mapping (x-1)
        wastes_arr_bins = np.array([sub_wastes[i] for i in range(1, len(sub_dist_matrix))])

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
