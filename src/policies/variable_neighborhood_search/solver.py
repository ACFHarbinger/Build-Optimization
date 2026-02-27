"""
Variable Neighborhood Search (VNS) for VRPP.

VNS systematically changes neighborhood structures to escape local optima.
Each outer iteration consists of two phases:

  1. Shaking: generate a random neighbour in the k-th shaking structure N_k
     (increasing severity as k grows).
  2. Local search descent: apply repeated LLH improvement from the shaken
     solution until no further improvement within the budget.

If the local search result improves the incumbent, the algorithm resets to
the first (mildest) neighborhood k=1.  Otherwise it advances to k+1.  Once
all k_max shaking structures are exhausted without improvement one outer
iteration is complete.  The process repeats for max_iterations outer loops
or until the time_limit is reached.

Reference:
    Hansen & Mladenović, "Variable Neighborhood Search", 1999.
"""

import copy
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.tracking.viz_mixin import PolicyVizMixin

from ..operators.destroy_operators import cluster_removal, random_removal, worst_removal
from ..operators.repair_operators import greedy_insertion, regret_2_insertion
from .params import VNSParams


class VNSSolver(PolicyVizMixin):
    """
    Variable Neighborhood Search solver for VRPP.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: VNSParams,
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

        # Shaking neighborhoods N_1 ... N_{k_max} ordered by increasing severity
        self._neighborhoods = [
            self._shake_n1,
            self._shake_n2,
            self._shake_n3,
            self._shake_n4,
            self._shake_n5,
        ]

        # LLH pool for local search descent phase
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
        Run Variable Neighborhood Search.

        Returns:
            Tuple of (routes, profit, cost).
        """
        if self.n_nodes == 0:
            return [], 0.0, 0.0

        start = time.time()
        k_max = min(self.params.k_max, len(self._neighborhoods))

        routes = self._build_initial_solution()
        profit = self._evaluate(routes)
        best_routes = copy.deepcopy(routes)
        best_profit = profit

        for iteration in range(self.params.max_iterations):
            if time.time() - start > self.params.time_limit:
                break

            k = 0  # 0-based index into self._neighborhoods
            while k < k_max:
                if time.time() - start > self.params.time_limit:
                    break

                # === Shaking phase ===
                try:
                    shaken = self._neighborhoods[k](copy.deepcopy(routes))
                except Exception:
                    k += 1
                    continue

                # === Local search descent phase ===
                ls_routes, ls_profit = self._local_search(shaken, start)

                # === Move or not (VNS acceptance criterion) ===
                if ls_profit > profit:
                    routes = ls_routes
                    profit = ls_profit
                    if profit > best_profit:
                        best_routes = copy.deepcopy(routes)
                        best_profit = profit
                    k = 0  # Improvement: restart from the mildest neighborhood
                else:
                    k += 1  # No improvement: try next neighborhood

            self._viz_record(
                iteration=iteration,
                best_profit=best_profit,
                best_cost=self._cost(best_routes),
            )

        best_cost = self._cost(best_routes)
        return best_routes, best_profit, best_cost

    # ------------------------------------------------------------------
    # Shaking neighborhoods (N_1 ... N_5, increasing severity)
    # ------------------------------------------------------------------

    def _shake_n1(self, routes: List[List[int]]) -> List[List[int]]:
        """N_1: Remove 1 node randomly, greedy reinsert."""
        partial, removed = random_removal(routes, 1)
        return greedy_insertion(
            partial,
            removed,
            self.dist_matrix,
            self.wastes,
            self.capacity,
            R=self.R,
            mandatory_nodes=self.mandatory_nodes,
        )

    def _shake_n2(self, routes: List[List[int]]) -> List[List[int]]:
        """N_2: Remove 2 nodes randomly, greedy reinsert."""
        partial, removed = random_removal(routes, 2)
        return greedy_insertion(
            partial,
            removed,
            self.dist_matrix,
            self.wastes,
            self.capacity,
            R=self.R,
            mandatory_nodes=self.mandatory_nodes,
        )

    def _shake_n3(self, routes: List[List[int]]) -> List[List[int]]:
        """N_3: Worst removal of 2 nodes, regret-2 reinsert."""
        partial, removed = worst_removal(routes, 2, self.dist_matrix)
        return regret_2_insertion(
            partial,
            removed,
            self.dist_matrix,
            self.wastes,
            self.capacity,
            R=self.R,
            mandatory_nodes=self.mandatory_nodes,
        )

    def _shake_n4(self, routes: List[List[int]]) -> List[List[int]]:
        """N_4: Cluster removal of 3 nodes, greedy reinsert."""
        partial, removed = cluster_removal(routes, 3, self.dist_matrix, self.nodes)
        return greedy_insertion(
            partial,
            removed,
            self.dist_matrix,
            self.wastes,
            self.capacity,
            R=self.R,
            mandatory_nodes=self.mandatory_nodes,
        )

    def _shake_n5(self, routes: List[List[int]]) -> List[List[int]]:
        """N_5: Remove 3 nodes randomly, regret-2 reinsert."""
        partial, removed = random_removal(routes, 3)
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
    # Local search descent
    # ------------------------------------------------------------------

    def _local_search(
        self,
        routes: List[List[int]],
        start: float,
    ) -> Tuple[List[List[int]], float]:
        """
        Apply repeated LLH improvement until no further progress or budget reached.

        Args:
            routes: Starting solution for the descent.
            start: Wall-clock start time of the outer solve() call.

        Returns:
            (routes, profit) after descent.
        """
        profit = self._evaluate(routes)

        for _ in range(self.params.local_search_iterations):
            if time.time() - start > self.params.time_limit:
                break

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

        return routes, profit

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
