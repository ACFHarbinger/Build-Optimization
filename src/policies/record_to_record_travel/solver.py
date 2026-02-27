"""
Record-to-Record Travel (RR) for VRPP.

The algorithm tracks the best solution found so far (the "record") and accepts
a candidate solution if its deviation from the record does not exceed a
predefined tolerance threshold.  The tolerance decays linearly over iterations,
providing a smooth transition from exploration to exploitation.

Reference:
    Dueck & Scheuer, "Threshold Accepting: A General Purpose Optimization
    Algorithm Appearing Superior to Simulated Annealing", 1990.
"""

import copy
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.tracking.viz_mixin import PolicyVizMixin

from ..operators.destroy_operators import cluster_removal, random_removal, worst_removal
from ..operators.repair_operators import greedy_insertion, regret_2_insertion
from .params import RRParams


class RRSolver(PolicyVizMixin):
    """
    Record-to-Record Travel solver for VRPP.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: RRParams,
        mandatory_nodes: Optional[List[int]] = None,
    ):
        self.dist_matrix = dist_matrix
        self.wastes = wastes
        self.capacity = capacity
        self.R = R
        self.C = C
        self.params = params
        self.mandatory_nodes = mandatory_nodes or []
        self.n_nodes = len(dist_matrix) - 1
        self.nodes = list(range(1, self.n_nodes + 1))

        self._llh_pool = [
            self._llh0,
            self._llh1,
            self._llh2,
            self._llh3,
            self._llh4,
        ]

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def solve(self) -> Tuple[List[List[int]], float, float]:
        """
        Run Record-to-Record Travel optimisation.

        Returns:
            Tuple of (routes, profit, cost).
        """
        if self.n_nodes == 0:
            return [], 0.0, 0.0

        start = time.time()

        # Initial solution
        routes = self._build_initial_solution()
        profit = self._evaluate(routes)

        record_routes = copy.deepcopy(routes)
        record_profit = profit

        # Tolerance band decays linearly over iterations
        initial_tolerance = self.params.tolerance * max(abs(record_profit), 1.0)

        for iteration in range(self.params.max_iterations):
            if time.time() - start > self.params.time_limit:
                break

            # Linear decay
            progress = iteration / max(self.params.max_iterations - 1, 1)
            tolerance = initial_tolerance * (1.0 - progress)

            # Select and apply a random LLH
            llh_idx = random.randint(0, self.params.n_llh - 1)
            llh = self._llh_pool[llh_idx]

            try:
                new_routes = llh(copy.deepcopy(routes), self.params.n_removal)
                new_profit = self._evaluate(new_routes)
            except Exception:
                continue

            # RR acceptance: accept if within tolerance of the record
            if new_profit >= record_profit - tolerance:
                routes = new_routes
                profit = new_profit

                # Update record
                if profit > record_profit:
                    record_routes = copy.deepcopy(routes)
                    record_profit = profit

            self._viz_record(
                iteration=iteration,
                best_profit=record_profit,
                best_cost=self._cost(record_routes),
                tolerance=tolerance,
            )

        record_cost = self._cost(record_routes)
        return record_routes, record_profit, record_cost

    # ------------------------------------------------------------------
    # LLH pool
    # ------------------------------------------------------------------

    def _llh0(self, routes: List[List[int]], n: int) -> List[List[int]]:
        partial, removed = random_removal(routes, n)
        return greedy_insertion(
            partial,
            removed,
            self.dist_matrix,
            self.wastes,
            self.capacity,
            R=self.R,
            mandatory_nodes=self.mandatory_nodes,
        )

    def _llh1(self, routes: List[List[int]], n: int) -> List[List[int]]:
        partial, removed = worst_removal(routes, n, self.dist_matrix)
        return regret_2_insertion(
            partial,
            removed,
            self.dist_matrix,
            self.wastes,
            self.capacity,
            R=self.R,
            mandatory_nodes=self.mandatory_nodes,
        )

    def _llh2(self, routes: List[List[int]], n: int) -> List[List[int]]:
        partial, removed = cluster_removal(routes, n, self.dist_matrix, self.nodes)
        return greedy_insertion(
            partial,
            removed,
            self.dist_matrix,
            self.wastes,
            self.capacity,
            R=self.R,
            mandatory_nodes=self.mandatory_nodes,
        )

    def _llh3(self, routes: List[List[int]], n: int) -> List[List[int]]:
        partial, removed = worst_removal(routes, n, self.dist_matrix)
        return greedy_insertion(
            partial,
            removed,
            self.dist_matrix,
            self.wastes,
            self.capacity,
            R=self.R,
            mandatory_nodes=self.mandatory_nodes,
        )

    def _llh4(self, routes: List[List[int]], n: int) -> List[List[int]]:
        partial, removed = random_removal(routes, n)
        return regret_2_insertion(
            partial,
            removed,
            self.dist_matrix,
            self.wastes,
            self.capacity,
            R=self.R,
            mandatory_nodes=self.mandatory_nodes,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_initial_solution(self) -> List[List[int]]:
        from logic.src.policies.operators.heuristics.initialization import build_nn_routes

        routes = build_nn_routes(
            nodes=self.nodes,
            mandatory_nodes=self.mandatory_nodes,
            wastes=self.wastes,
            capacity=self.capacity,
            dist_matrix=self.dist_matrix,
            R=self.R,
            C=self.C,
        )

        return routes

    def _evaluate(self, routes: List[List[int]]) -> float:
        if not routes:
            return 0.0
        rev = sum(self.wastes.get(n, 0.0) * self.R for r in routes for n in r)
        return rev - self._cost(routes) * self.C

    def _cost(self, routes: List[List[int]]) -> float:
        total = 0.0
        for route in routes:
            if not route:
                continue
            total += self.dist_matrix[0][route[0]]
            for k in range(len(route) - 1):
                total += self.dist_matrix[route[k]][route[k + 1]]
            total += self.dist_matrix[route[-1]][0]
        return total
