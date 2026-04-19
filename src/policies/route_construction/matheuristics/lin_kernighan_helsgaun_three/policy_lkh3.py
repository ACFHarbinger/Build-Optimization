"""
LKH-3 Policy Adapter.

Adapts the Lin-Kernighan-Helsgaun 3 heuristic to the agnostic routing
policy interface used by the WSmart+ Route simulator.

Reference:
    Helsgaun, K. (2017). An extension of the LKH-TSP solver for constrained
    traveling salesman and vehicle routing problems.
"""

import time
from random import Random
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies.lkh3 import LKH3Config
from logic.src.enums import GlobalRegistry, PolicyTag
from logic.src.policies.route_construction.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.route_construction.base.factory import RouteConstructorRegistry
from logic.src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.graph_augmentation import (
    augment_graph,
)
from logic.src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.lkh3 import (
    solve_lkh3,
    solve_lkh3_with_alns,
)

from .params import LKH3Params


@GlobalRegistry.register(
    PolicyTag.MATHEURISTIC,
    PolicyTag.META_HEURISTIC,
    PolicyTag.LOCAL_SEARCH,
    PolicyTag.PROFIT_AWARE,
)
@RouteConstructorRegistry.register("lkh3")
class LKH3Policy(BaseRoutingPolicy):
    """LKH-3 policy class.

    Solves the CVRP/TSP sub-problem via the Lin-Kernighan-Helsgaun 3
    iterated local-search heuristic with POPMUSIC candidate generation,
    k-opt moves (k = 2..5), and IP-based tour merging.
    """

    def __init__(self, config: Optional[Union[LKH3Config, Dict[str, Any]]] = None):
        """Initialize LKH-3 policy with optional config.

        Args:
            config: LKH3Config dataclass, raw dict from YAML, or None.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return LKH3Config

    def _get_config_key(self) -> str:
        """Return config key for LKH-3."""
        return "lkh3"

    def _tour_to_routes(self, tour: List[int]) -> List[List[int]]:
        """Split a flat closed tour at depot (0) into individual route lists.

        Args:
            tour: Flat tour with depot visits (e.g. ``[0, 3, 1, 0, 2, 4, 0]``).

        Returns:
            List of route lists, each without leading/trailing depot.
            Example: ``[[3, 1], [2, 4]]``.
        """
        routes: List[List[int]] = []
        current: List[int] = []
        for node in tour:
            if node == 0:
                if current:
                    routes.append(current)
                    current = []
            else:
                current.append(node)
        if current:
            routes.append(current)
        return routes

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
        Execute the Lin-Kernighan-Helsgaun 3 (LKH-3) solver logic.

        LKH-3 is an extremely powerful iterated local search heuristic. It generalizes
        the classic K-opt Lin-Kernighan move for various constrained TSP and VRP
        variants. This implementation specifically supports CVRP and VRPP
        through graph augmentation (dummy nodes/edges) and can use ALNS for
        intensification within the LKH framework.

        Args:
            sub_dist_matrix (np.ndarray): Symmetric distance matrix (local indices).
            sub_wastes (Dict[int, float]): Mapping of local node indices to
                current bin inventory levels.
            capacity (float): Maximum vehicle collection capacity.
            revenue (float): Revenue obtained per kilogram of waste collected.
            cost_unit (float): Monetary cost incurred per kilometer traveled.
            values (Dict[str, Any]): Merged configuration dictionary containing
                LKH-3 parameters (runs, max_trials, max_k_opt, etc.) and ALNS settings.
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
        n_original = len(sub_dist_matrix)

        # --- Initialize type-safe Params ---
        params = LKH3Params.from_config(self._config or values)
        seed = params.seed
        n_vehicles = int(values.get("n_vehicles", 3))  # Fleet size comes from simulation kwargs

        np_rng = np.random.default_rng(seed if seed is not None else 42)

        # --- Phase 1: Graph Augmentation ---
        # If native prize-collecting is enabled, we skip standard CVRP augmentation
        # as the solver will perform Jonker-Volgenant ATSP transformation internally.
        # Select LKH-3 variant (with or without Adaptive Large Neighborhood Search)
        policy_function = solve_lkh3_with_alns if params.vrpp or params.native_prize_collecting else solve_lkh3

        # Type-safe demand representation
        waste_arr: Optional[np.ndarray] = None
        if params.native_prize_collecting:
            augmented_dist = sub_dist_matrix
            waste_arr = np.zeros(len(sub_dist_matrix))
            for i, w in sub_wastes.items():
                waste_arr[i] = w
            n_original = len(sub_dist_matrix)
        else:
            augmented_dist, augmented_waste, n_original = augment_graph(
                distance_matrix=sub_dist_matrix,
                wastes=sub_wastes,
                n_vehicles=n_vehicles,
                capacity=capacity,
            )
            waste_arr = augmented_waste if capacity > 0 else None

        cap: float = capacity if capacity > 0 else 100.0

        # --- Run the LKH-3 engine on augmented graph ---
        best_routes: Optional[List[List[int]]] = None
        best_cost = float("inf")
        rng = Random(seed)
        start_time = time.process_time()
        for _ in range(params.runs):
            current_time = time.process_time()
            if params.time_limit > 0 and (current_time - start_time) > params.time_limit:
                break
            routes, cost, _ = policy_function(
                distance_matrix=augmented_dist,
                initial_tour=None,
                waste=waste_arr,
                capacity=cap,
                revenue=revenue,
                cost_unit=cost_unit,
                mandatory_nodes=mandatory_nodes,
                coords=sub_dist_matrix,
                max_trials=params.max_trials,
                popmusic_subpath_size=params.popmusic_subpath_size,
                popmusic_trials=params.popmusic_trials,
                popmusic_max_candidates=params.popmusic_max_candidates,
                max_k_opt=params.max_k_opt,
                use_ip_merging=params.use_ip_merging,
                max_pool_size=params.max_pool_size,
                subgradient_iterations=params.subgradient_iterations,
                profit_aware_operators=params.profit_aware_operators,
                alns_iterations=params.alns_iterations,
                plateau_limit=params.plateau_limit,
                deep_plateau_limit=params.deep_plateau_limit,
                perturb_operator_weights=params.perturb_operator_weights,
                n_vehicles=n_vehicles,
                n_original=n_original,
                recorder=self._viz,
                np_rng=np_rng,
                rng=rng,
                seed=seed,
                dynamic_topology_discovery=params.dynamic_topology_discovery,
                native_prize_collecting=params.native_prize_collecting,
            )
            if cost < best_cost:
                best_cost = cost
                best_routes = routes

        if best_routes is None:
            # Fallback: depot-only tour
            return [[0]], 0.0, 0.0

        # --- Profit calculation (VRPP: revenue * collected − routing cost) ---
        collected_nodes = {node for route in best_routes for node in route}
        routing_cost = self._route_cost(best_routes, sub_dist_matrix)
        total_fill = sum(sub_wastes.get(node, 0.0) for node in collected_nodes)
        profit = revenue * total_fill - cost_unit * routing_cost
        return best_routes, profit, routing_cost

    # ------------------------------------------------------------------
    # Mandatory-node helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _route_cost(routes: List[List[int]], dist: np.ndarray) -> float:
        """Total routing distance for a set of routes (depot = 0).

        Args:
            routes: List of route lists (no depot).
            dist: Distance matrix.

        Returns:
            Total travel distance.
        """
        total = 0.0
        for route in routes:
            if not route:
                continue
            total += dist[0, route[0]]
            for k in range(len(route) - 1):
                total += dist[route[k], route[k + 1]]
            total += dist[route[-1], 0]
        return total

    def _enforce_mandatory_and_filter(
        self,
        routes: List[List[int]],
        mandatory_set: set,
        dist: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        revenue: float,
        cost_unit: float,
    ) -> List[List[int]]:
        """Enforce mandatory nodes and filter unprofitable optional nodes.

        1. Removes optional (non-mandatory) nodes whose marginal profit
           contribution is negative — i.e., the cost of the detour to
           collect them exceeds the revenue from their fill level.
        2. Ensures every mandatory node appears in at least one route;
           any missing mandatory node is appended as a single-node route.

        Args:
            routes: Current routes (from the LKH-3 tour split).
            mandatory_set: Set of local node indices that must be visited.
            dist: (N × N) distance matrix.
            wastes: ``{local_idx: fill_level}``.
            capacity: Vehicle capacity.
            revenue: Revenue per unit collected.
            cost_unit: Cost per distance unit.

        Returns:
            Filtered routes with mandatory nodes guaranteed.
        """
        filtered_routes: List[List[int]] = []
        for route in routes:
            if not route:
                continue

            # Keep mandatory nodes unconditionally; filter optional ones
            kept: List[int] = []
            for node in route:
                if node in mandatory_set:
                    kept.append(node)
                else:
                    # Check marginal profit of visiting this optional node
                    node_fill = wastes.get(node, 0.0)
                    node_revenue = revenue * node_fill

                    # Estimate detour cost: cost of inserting this node
                    # between its predecessor and successor
                    prev = kept[-1] if kept else 0
                    # Approximate successor as depot (worst case)
                    detour_cost = cost_unit * (dist[prev, node] + dist[node, 0] - dist[prev, 0])

                    if node_revenue > detour_cost:
                        kept.append(node)

            if kept:
                # Re-check capacity feasibility
                route_load = sum(wastes.get(n, 0.0) for n in kept)
                if route_load <= capacity + 1e-6:
                    filtered_routes.append(kept)
                # NOTE: No post-hoc capacity splitting! The LKH-3 engine with
                # augmented dummy depots handles multi-route optimization natively.
                # If a filtered route exceeds capacity, it's discarded rather than
                # split, as splitting would destroy the LKH-3 optimization.

        # Ensure every mandatory node is visited
        visited = {node for route in filtered_routes for node in route}
        missing = mandatory_set - visited
        for node in sorted(missing):
            filtered_routes.append([node])

        return filtered_routes
