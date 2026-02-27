"""
Guided Local Search (GLS) for VRPP.

GLS augments the objective function with adaptive penalty terms on routing
edge features that appear in local optima.  When the inner local search
stagnates, the feature with the highest utility is penalised, artificially
inflating the cost of the current optimum and guiding the search toward
unexplored basins.

Reference:
    Voudouris & Tsang, "Guided Local Search and Its Application to the
    Traveling Salesman Problem", 1999.
"""

import copy
import random
import time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from logic.src.tracking.viz_mixin import PolicyVizMixin

from ..operators.destroy_operators import cluster_removal, random_removal, worst_removal
from ..operators.repair_operators import greedy_insertion, regret_2_insertion
from .params import GLSParams


class GLSSolver(PolicyVizMixin):
    """
    Guided Local Search solver for VRPP.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: GLSParams,
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

        # Edge penalty matrix (features = edges)
        n = len(dist_matrix)
        self.penalties = np.zeros((n, n), dtype=np.float64)

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
        Run GLS optimisation.

        Returns:
            Tuple of (routes, profit, cost).
        """
        if self.n_nodes == 0:
            return [], 0.0, 0.0

        start = time.time()

        routes = self._build_initial_solution()
        profit = self._evaluate(routes)
        best_routes = copy.deepcopy(routes)
        best_profit = profit

        for restart in range(self.params.max_restarts):
            if time.time() - start > self.params.time_limit:
                break

            # Inner local search loop using augmented objective
            for _ in range(self.params.inner_iterations):
                if time.time() - start > self.params.time_limit:
                    break

                llh_idx = random.randint(0, self.params.n_llh - 1)
                llh = self._llh_pool[llh_idx]

                try:
                    new_routes = llh(copy.deepcopy(routes), self.params.n_removal)
                except Exception:
                    continue

                # Accept if augmented objective improves
                aug_new = self._augmented_evaluate(new_routes)
                aug_cur = self._augmented_evaluate(routes)

                if aug_new >= aug_cur:
                    routes = new_routes
                    real_profit = self._evaluate(routes)

                    if real_profit > best_profit:
                        best_routes = copy.deepcopy(routes)
                        best_profit = real_profit

            # At local optimum: penalise the edge with highest utility
            self._update_penalties(routes)

            self._viz_record(
                iteration=restart,
                best_profit=best_profit,
                best_cost=self._cost(best_routes),
            )

        best_cost = self._cost(best_routes)
        return best_routes, best_profit, best_cost

    # ------------------------------------------------------------------
    # Penalty management
    # ------------------------------------------------------------------

    def _get_edges(self, routes: List[List[int]]) -> Set[Tuple[int, int]]:
        """Extract all edges from routes (including depot connections)."""
        edges: Set[Tuple[int, int]] = set()
        for route in routes:
            if not route:
                continue
            edges.add((0, route[0]))
            for k in range(len(route) - 1):
                edges.add((route[k], route[k + 1]))
            edges.add((route[-1], 0))
        return edges

    def _update_penalties(self, routes: List[List[int]]) -> None:
        """Penalise the edge feature with highest utility."""
        edges = self._get_edges(routes)
        if not edges:
            return

        best_utility = -1.0
        best_edge = None

        for i, j in edges:
            cost_ij = self.dist_matrix[i][j]
            utility = cost_ij / (1.0 + self.penalties[i][j])
            if utility > best_utility:
                best_utility = utility
                best_edge = (i, j)

        if best_edge is not None:
            self.penalties[best_edge[0]][best_edge[1]] += 1.0

    def _augmented_evaluate(self, routes: List[List[int]]) -> float:
        """Evaluate with penalty-augmented objective."""
        real = self._evaluate(routes)
        penalty = 0.0
        for route in routes:
            if not route:
                continue
            penalty += self.penalties[0][route[0]]
            for k in range(len(route) - 1):
                penalty += self.penalties[route[k]][route[k + 1]]
            penalty += self.penalties[route[-1]][0]
        return real - self.params.lambda_param * penalty

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
