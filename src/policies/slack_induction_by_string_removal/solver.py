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
"""

import copy
import math
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.tracking.viz_mixin import PolicyVizMixin

from ..operators.destroy_operators import string_removal
from ..operators.repair_operators import greedy_insertion_with_blinks
from .params import SISRParams


class SISRSolver(PolicyVizMixin):
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
    ):
        """
        Initialize SISR Solver.

        Args:
            params: Parameters for the SISR algorithm.
        """
        self.dist_matrix = dist_matrix
        self.wastes = wastes
        self.capacity = capacity
        self.R = R
        self.C = C
        self.params = params

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
        start_time = time.time()

        n_nodes = len(self.dist_matrix) - 1
        n_remove = max(1, int(n_nodes * self.params.destroy_ratio))

        for _it in range(self.params.max_iterations):
            if time.time() - start_time > self.params.time_limit:
                break

            # SISR Iteration
            # 1. Destroy
            partial_routes, removed = string_removal(
                copy.deepcopy(current_routes),
                n_remove,
                self.dist_matrix,
                max_string_len=self.params.max_string_len,
                avg_string_len=self.params.avg_string_len,
            )

            # 2. Repair
            new_routes = greedy_insertion_with_blinks(
                partial_routes,
                removed,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                blink_rate=self.params.blink_rate,
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
                if random.random() < prob:
                    accept = True

            if accept:
                current_routes = new_routes
                current_cost = new_profit
                if current_cost > best_cost + 1e-6:
                    best_routes = [r[:] for r in current_routes]
                    best_cost = current_cost

            # Cooling
            T *= self.params.cooling_rate

            self._viz_record(
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
        from logic.src.policies.operators.heuristics.initialization import build_nn_routes

        routes = build_nn_routes(
            nodes=list(self.wastes.keys()),
            mandatory_nodes=[],  # SISR doesn't use mandatory_nodes parameter
            wastes=self.wastes,
            capacity=self.capacity,
            dist_matrix=self.dist_matrix,
            R=self.R,
            C=self.C,
        )
        return routes
