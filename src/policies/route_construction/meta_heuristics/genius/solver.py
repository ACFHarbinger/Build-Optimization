"""
GENIUS (GENI + US) Meta-Heuristic for VRPP.

Implements the GENIUS algorithm combining:
- GENI (Generalized Insertion): Construction heuristic with Type I and Type II moves
- US (Unstringing and Stringing): Post-optimization procedure

Reference:
    Gendreau, M., Hertz, A., & Laporte, G. (1992).
    "New Insertion and Postoptimization Procedures for the Traveling Salesman Problem"
    Operations Research, 40(6), 1086-1094.
"""

import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.policies.helpers.operators.generalized_insertion_and_deletion.stringing_wrapper import (
    stringing_insertion,
    stringing_profit_insertion,
)
from logic.src.policies.helpers.operators.generalized_insertion_and_deletion.unstringing_wrapper import (
    unstringing_profit_removal,
    unstringing_removal,
)
from logic.src.policies.helpers.operators.recreate_repair.geni import geni_insertion, geni_profit_insertion

from .params import GENIUSParams


class GENIUSSolver:
    """
    GENIUS meta-heuristic solver for VRPP.

    Combines GENI insertion with Unstringing/Stringing post-optimization.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: GENIUSParams,
        mandatory_nodes: Optional[List[int]] = None,
    ):
        """
        Initialize the GENIUS solver.

        Args:
            dist_matrix: Distance matrix (n_nodes+1 x n_nodes+1), index 0 is depot.
            wastes: Dictionary mapping node IDs to waste amounts (demands).
            capacity: Vehicle capacity constraint.
            R: Revenue multiplier for profit calculation.
            C: Cost multiplier for profit calculation.
            params: GENIUS algorithm parameters.
            mandatory_nodes: List of nodes that must be visited.
        """
        self.dist_matrix = dist_matrix
        self.wastes = wastes
        self.capacity = capacity
        self.R = R
        self.C = C
        self.params = params
        self.mandatory_nodes = mandatory_nodes or []
        self.n_nodes = len(dist_matrix) - 1
        self.nodes = list(range(1, self.n_nodes + 1))
        self.random = random.Random(params.seed) if params.seed is not None else random.Random()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def solve(self) -> Tuple[List[List[int]], float, float]:
        """
        Run the GENIUS algorithm with corrected US acceptance criterion.

        The US procedure follows Algorithm US from Gendreau et al. (1992):
        1. Track both current_routes (working solution) and best_routes (global optimum)
        2. Accept EVERY US move on a node (update current_routes), even if it worsens profit
        3. Only update best_routes when current_profit > best_profit
        4. Terminate when a full sweep through all nodes produces no new global best

        Returns:
            Tuple of (routes, profit, cost).
                routes: List of routes, each route is a list of node IDs.
                profit: Total profit (revenue - cost).
                cost: Total routing cost.
        """
        if self.n_nodes == 0:
            return [], 0.0, 0.0

        start = time.process_time()

        # Step 1: Initial solution using GENI insertion
        routes = self._build_initial_solution_geni()
        profit = self._evaluate(routes)

        # Initialize best solution (global optimum)
        best_routes = [r[:] for r in routes]
        best_profit = profit

        # Initialize current solution (working solution for US)
        current_routes = [r[:] for r in routes]
        current_profit = profit

        # Step 2: Iterative improvement using US (Unstringing-Stringing)
        for iteration in range(self.params.n_iterations):
            if self.params.time_limit > 0 and time.process_time() - start > self.params.time_limit:
                break

            # Apply US sweep: Continue until a full sweep produces no global improvement
            while True:
                if self.params.time_limit > 0 and time.process_time() - start > self.params.time_limit:
                    break

                global_improvement_found = False
                current_nodes = [node for r in current_routes for node in r]

                # Attempt US on every node in the current working solution
                for node_to_remove in current_nodes:
                    if self.params.time_limit > 0 and time.process_time() - start > self.params.time_limit:
                        break

                    # Unstringing phase: Remove exactly one node
                    routes_copy = [r[:] for r in current_routes]

                    try:
                        # Phase 1: Patch Node Loss Bug - Record count BEFORE US
                        original_node_count = sum(len(r) for r in current_routes)

                        if self.params.profit_aware_operators:
                            routes_us, removed = unstringing_profit_removal(
                                routes=routes_copy,
                                n_remove=1,
                                unstring_type=self.params.unstring_type,
                                dist_matrix=self.dist_matrix,
                                wastes=self.wastes,
                                R=self.R,
                                C=self.C,
                                rng=self.random,
                                target_node=node_to_remove,
                                use_alns_fallback=False,
                                random_us_sampling=self.params.random_us_sampling,
                                neighborhood_size=self.params.neighborhood_size,
                            )
                        else:
                            routes_us, removed = unstringing_removal(
                                routes=routes_copy,
                                n_remove=1,
                                unstring_type=self.params.unstring_type,
                                dist_matrix=self.dist_matrix,
                                rng=self.random,
                                target_node=node_to_remove,
                                use_alns_fallback=False,
                                random_us_sampling=self.params.random_us_sampling,
                                neighborhood_size=self.params.neighborhood_size,
                            )

                        # Stringing phase: Reinsert removed node
                        if removed and removed[0] == node_to_remove:
                            if self.params.profit_aware_operators:
                                routes_us = stringing_profit_insertion(
                                    routes=routes_us,
                                    removed_nodes=removed,
                                    string_type=self.params.string_type,
                                    dist_matrix=self.dist_matrix,
                                    wastes=self.wastes,
                                    capacity=self.capacity,
                                    R=self.R,
                                    C=self.C,
                                    mandatory_nodes=self.mandatory_nodes,
                                    rng=self.random,
                                    expand_pool=False,
                                    use_alns_fallback=False,
                                    random_us_sampling=self.params.random_us_sampling,
                                    neighborhood_size=self.params.neighborhood_size,
                                )
                            else:
                                routes_us = stringing_insertion(
                                    routes=routes_us,
                                    removed_nodes=removed,
                                    string_type=self.params.string_type,
                                    dist_matrix=self.dist_matrix,
                                    wastes=self.wastes,
                                    capacity=self.capacity,
                                    mandatory_nodes=self.mandatory_nodes,
                                    rng=self.random,
                                    expand_pool=False,
                                    use_alns_fallback=False,
                                    random_us_sampling=self.params.random_us_sampling,
                                    neighborhood_size=self.params.neighborhood_size,
                                )

                            # Evaluate the US move
                            new_profit = self._evaluate(routes_us)

                            # Phase 1: Patch Node Loss Bug - Rejection Logic
                            new_node_count = sum(len(r) for r in routes_us)
                            if new_node_count < original_node_count:
                                continue

                            # CORRECTED ACCEPTANCE CRITERION:
                            # Always accept the US move (update current_routes)
                            # even if it worsens the profit. This allows escaping local optima.
                            current_routes = routes_us
                            current_profit = new_profit

                            # Only update global best if we found a better solution
                            if current_profit > best_profit:
                                best_routes = [r[:] for r in current_routes]
                                best_profit = current_profit
                                global_improvement_found = True

                    except Exception as e:
                        # If US fails on this node, continue to the next node
                        print(f"Warning: US Operator failed on node {node_to_remove} | Error: {e}")
                        continue

                # Termination: If we completed a full sweep without finding a new global best, stop
                if not global_improvement_found:
                    break

            # Record progress (for visualization/logging hooks)
            getattr(self, "_viz_record", lambda **k: None)(
                iteration=iteration,
                best_profit=best_profit,
                best_cost=self._cost(best_routes),
            )

        best_cost = self._cost(best_routes)
        return best_routes, best_profit, best_cost

    # ------------------------------------------------------------------
    # Initial solution construction
    # ------------------------------------------------------------------

    def _build_initial_solution_geni(self) -> List[List[int]]:
        """
        Build initial solution using GENI (Generalized Insertion).

        GENI considers Type I and Type II insertions with p-neighborhood restriction.
        Uses deterministic p-neighborhood search when random_us_sampling is False.

        Returns:
            List of routes constructed using GENI.
        """
        # Start with empty routes
        routes: List[List[int]] = []

        # All nodes start as unassigned
        all_nodes = set(self.nodes)

        # Determine if we use deterministic p-neighborhood (invert random_us_sampling)
        use_deterministic = not self.params.random_us_sampling

        # Use GENI insertion to build the solution
        if self.params.profit_aware_operators:
            routes = geni_profit_insertion(
                routes=routes,
                removed_nodes=list(all_nodes),
                dist_matrix=self.dist_matrix,
                wastes=self.wastes,
                capacity=self.capacity,
                R=self.R,
                C=self.C,
                neighborhood_size=self.params.neighborhood_size,
                mandatory_nodes=self.mandatory_nodes,
                rng=self.random,
                expand_pool=self.params.vrpp,
                use_deterministic_p_neighborhood=use_deterministic,
            )
        else:
            routes = geni_insertion(
                routes=routes,
                removed_nodes=list(all_nodes),
                dist_matrix=self.dist_matrix,
                wastes=self.wastes,
                capacity=self.capacity,
                neighborhood_size=self.params.neighborhood_size,
                mandatory_nodes=self.mandatory_nodes,
                rng=self.random,
                expand_pool=self.params.vrpp,
                use_deterministic_p_neighborhood=use_deterministic,
            )

        # Clean up empty routes
        routes = [r for r in routes if r]

        return routes

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _evaluate(self, routes: List[List[int]]) -> float:
        """
        Calculate profit of a solution.

        Profit = Revenue - Cost
        Revenue = sum of wastes collected * R
        Cost = total distance * C

        Args:
            routes: List of routes.

        Returns:
            Total profit.
        """
        revenue = sum(self.wastes.get(node, 0) for route in routes for node in route) * self.R
        cost = self._cost(routes)
        return revenue - cost

    def _cost(self, routes: List[List[int]]) -> float:
        """
        Calculate total routing cost.

        Args:
            routes: List of routes.

        Returns:
            Total cost (distance * C).
        """
        total_dist = 0.0
        for route in routes:
            if not route:
                continue
            # Depot to first node
            total_dist += self.dist_matrix[0, route[0]]
            # Consecutive nodes
            for i in range(len(route) - 1):
                total_dist += self.dist_matrix[route[i], route[i + 1]]
            # Last node to depot
            total_dist += self.dist_matrix[route[-1], 0]
        return total_dist * self.C
