"""
SISR Solver Module.

This module implements the Slack Induction by String Removal (SISR) metaheuristic.
It manages the iterative process of string removal (ruin) and greedy insertion
with blinks (recreate), accepted via Simulated Annealing.

Attributes:
    None

Example:
    >>> from logic.src.policies.slack_induction_by_string_removal.solver import SISRSolver
    >>> solver = SISRSolver(dist_matrix, wastes, ...)
    >>> result = solver.solve()

Reference:
    Christiaens, J., & Berghe, V. G. "Slack induction by string removal
    for the vehicle routing problem.", 2022.
"""

import copy
import math
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.policies.helpers.operators import (
    greedy_insertion_with_blinks,
    greedy_profit_insertion_with_blinks,
    string_removal,
)
from logic.src.policies.helpers.operators.solution_initialization.greedy_si import (
    build_greedy_routes,
)

from .params import SISRParams


class SISRSolver:
    """
    Solver implementing the SISR metaheuristic.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: SISRParams,
        mandatory_nodes: Optional[List[int]] = None,
    ):
        """
        Initialize SISR Solver.

        Args:
            dist_matrix: NxN distance matrix.
            wastes: Waste values for each node.
            capacity: Vehicle capacity.
            R: Revenue multiplier.
            C: Cost multiplier.
            params: Parameters for the SISR algorithm.
        """
        self.dist_matrix = dist_matrix
        self.wastes = wastes
        self.capacity = capacity
        self.R = R
        self.C = C
        self.params = params
        self.mandatory_nodes = mandatory_nodes if mandatory_nodes is not None else []
        self.random = random.Random(params.seed) if params.seed is not None else random.Random()

    def solve(self, initial_solution: Optional[List[List[int]]] = None) -> Tuple[List[List[int]], float, float]:
        """
        Run the SISR algorithm.
        """
        current_routes = [r[:] for r in initial_solution] if initial_solution else self._build_initial_solution()

        best_routes = [r[:] for r in current_routes]
        collected_revenue = sum(self.wastes.get(node, 0) * self.R for route in best_routes for node in route)
        best_cost = collected_revenue - (self._calculate_cost(best_routes) * self.C)  # Profit
        current_cost = best_cost

        T = self.params.start_temp
        start_time = time.process_time()

        n_nodes = len(self.dist_matrix) - 1
        n_remove = max(1, int(n_nodes * self.params.destroy_ratio))

        for _it in range(self.params.max_iterations):
            if self.params.time_limit > 0 and time.process_time() - start_time > self.params.time_limit:
                break

            # SISR Iteration
            # 1. Destroy
            partial_routes, removed = string_removal(
                routes=copy.deepcopy(current_routes),
                n_remove=n_remove,
                dist_matrix=self.dist_matrix,
                max_string_len=self.params.max_string_len,
                avg_string_len=self.params.avg_string_len,
                rng=self.random,
            )

            # 2. Repair
            if self.params.profit_aware_operators:
                new_routes = greedy_profit_insertion_with_blinks(
                    routes=partial_routes,
                    removed_nodes=removed,
                    dist_matrix=self.dist_matrix,
                    wastes=self.wastes,
                    capacity=self.capacity,
                    R=self.R,
                    C=self.C,
                    blink_rate=self.params.blink_rate,
                    mandatory_nodes=self.mandatory_nodes,
                    rng=self.random,
                    expand_pool=self.params.vrpp,
                )
            else:
                new_routes = greedy_insertion_with_blinks(
                    routes=partial_routes,
                    removed_nodes=removed,
                    dist_matrix=self.dist_matrix,
                    wastes=self.wastes,
                    capacity=self.capacity,
                    blink_rate=self.params.blink_rate,
                    mandatory_nodes=self.mandatory_nodes,
                    rng=self.random,
                    expand_pool=self.params.vrpp,
                )

            # Calculate profit instead of cost for maximizing
            collected_revenue = sum(self.wastes.get(node, 0) * self.R for route in new_routes for node in route)
            new_cost = self._calculate_cost(new_routes)
            new_profit = collected_revenue - (new_cost * self.C)

            # Acceptance (Simulated Annealing) for maximization
            delta = new_profit - current_cost  # Notice: we compare profit
            accept = False

            if delta > -1e-6:
                accept = True
            else:
                prob = math.exp(delta / T) if T > 0 else 0
                if self.random.random() < prob:
                    accept = True

            if accept:
                current_routes = new_routes
                current_cost = new_profit
                if current_cost > best_cost + 1e-6:
                    best_routes = [r[:] for r in current_routes]
                    best_cost = current_cost

            # Cooling
            T *= self.params.cooling_rate

            getattr(self, "_viz_record", lambda **k: None)(
                iteration=_it,
                best_cost=best_cost,
                current_cost=current_cost,
                temperature=T,
                accepted=int(accept),
            )

        return best_routes, best_cost, self._calculate_cost(best_routes)

    def _calculate_cost(self, routes: List[List[int]]) -> float:
        total = 0.0
        for route in routes:
            if not route:
                continue
            total += float(self.dist_matrix[0, route[0]])
            for i in range(len(route) - 1):
                total += float(self.dist_matrix[route[i], route[i + 1]])
            total += float(self.dist_matrix[route[-1], 0])
        return total

    def _build_initial_solution(self) -> List[List[int]]:
        """Greedy constructive heuristic."""
        routes = build_greedy_routes(
            dist_matrix=self.dist_matrix,
            wastes=self.wastes,
            capacity=self.capacity,
            R=self.R,
            C=self.C,
            mandatory_nodes=self.mandatory_nodes,
            rng=self.random,
        )
        return routes

    def _calculate_profit(self, routes: List[List[int]]) -> float:
        """
        Calculate total profit (revenue - cost) for routes.

        Args:
            routes: List of routes.

        Returns:
            Total profit.
        """
        collected_revenue = sum(self.wastes.get(node, 0) * self.R for route in routes for node in route)
        cost = self._calculate_cost(routes) * self.C
        return collected_revenue - cost

    def _is_valid_route(self, route: List[int]) -> bool:
        """
        Check if a route is valid (respects capacity constraints).

        Args:
            route: Route to validate.

        Returns:
            True if valid, False otherwise.
        """
        if not route:
            return True

        # Check capacity constraint
        total_waste = sum(self.wastes.get(node, 0) for node in route)
        if total_waste > self.capacity:
            return False

        # Check for duplicate nodes (except depot)
        non_depot_nodes = [n for n in route if n != 0]
        return len(non_depot_nodes) == len(set(non_depot_nodes))
