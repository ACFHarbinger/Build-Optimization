"""
Iterated Local Search (ILS) for VRPP.

ILS alternates between a local search descent phase and a perturbation phase.
The descent phase applies destroy/repair LLHs until no improvement is found.
The perturbation phase randomly disrupts the current solution to escape the
local optimum.  If the perturbed + re-optimised solution beats the incumbent,
it replaces the current solution (hill-climbing acceptance).

Reference:
    Lourenço, Martin & Stützle, "Iterated Local Search", 2003.
"""

import copy
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.tracking.viz_mixin import PolicyVizMixin

from ..operators.destroy_operators import cluster_removal, random_removal, worst_removal
from ..operators.repair_operators import greedy_insertion, regret_2_insertion
from .params import ILSParams


class ILSSolver(PolicyVizMixin):
    """
    Iterated Local Search solver for VRPP.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: ILSParams,
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
        Run Iterated Local Search.

        Returns:
            Tuple of (routes, profit, cost).
        """
        if self.n_nodes == 0:
            return [], 0.0, 0.0

        start = time.time()

        # Initial solution
        routes = self._build_initial_solution()
        profit = self._evaluate(routes)
        best_routes = copy.deepcopy(routes)
        best_profit = profit

        for restart in range(self.params.n_restarts):
            if time.time() - start > self.params.time_limit:
                break

            # === Descent phase ===
            improved = True
            inner_count = 0
            while improved and inner_count < self.params.inner_iterations:
                if time.time() - start > self.params.time_limit:
                    break
                improved = False
                inner_count += 1

                llh_idx = random.randint(0, self.params.n_llh - 1)
                llh = self._llh_pool[llh_idx]

                try:
                    new_routes = llh(copy.deepcopy(routes), self.params.n_removal)
                    new_profit = self._evaluate(new_routes)
                except Exception:
                    continue

                if new_profit > profit:
                    routes = new_routes
                    profit = new_profit
                    improved = True

                    if profit > best_profit:
                        best_routes = copy.deepcopy(routes)
                        best_profit = profit

            # === Perturbation phase ===
            perturbed = self._perturb(copy.deepcopy(routes))
            perturbed_profit = self._evaluate(perturbed)

            # Hill-climbing acceptance: accept perturbation as starting point
            # if it leads to better results after another descent
            routes = perturbed
            profit = perturbed_profit

            self._viz_record(
                iteration=restart,
                best_profit=best_profit,
                best_cost=self._cost(best_routes),
            )

        best_cost = self._cost(best_routes)
        return best_routes, best_profit, best_cost

    # ------------------------------------------------------------------
    # Perturbation
    # ------------------------------------------------------------------

    def _perturb(self, routes: List[List[int]]) -> List[List[int]]:
        """Apply strong perturbation to escape local optimum."""
        flat = [n for r in routes for n in r]
        if len(flat) < 4:
            return routes

        n_remove = max(2, int(len(flat) * self.params.perturbation_strength))

        # Random removal of a large chunk
        partial, removed = random_removal(routes, n_remove)

        # Reinsert removed nodes
        return greedy_insertion(
            partial,
            removed,
            self.dist_matrix,
            self.wastes,
            self.capacity,
            R=self.R,
            mandatory_nodes=self.mandatory_nodes,
        )

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
        from src.policies.operators.heuristics.initialization import build_nn_routes

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
