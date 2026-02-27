"""
Sine Cosine Algorithm (SCA) for VRPP.

Positions are real-valued vectors updated each iteration by trigonometric
wave functions.  Exploration / exploitation balance is enforced by the
parameter `a` which decays linearly from `a_max` to 0 over the run.  The
final continuous vector is binarised via a sigmoid transfer function and
decoded to a discrete routing solution via the Largest Rank Value (LRV) rule.

Reference:
    Survey §"Sine Cosine Algorithm" — multi-group adaptation for CVRP/VRPP.
"""

import copy
import math
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.tracking.viz_mixin import PolicyVizMixin

from .params import SCAParams


class SCASolver(PolicyVizMixin):
    """
    Sine Cosine Algorithm solver for VRPP.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: SCAParams,
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

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def solve(self) -> Tuple[List[List[int]], float, float]:
        """
        Run SCA and return the best solution found.

        Returns:
            Tuple of (routes, profit, cost).
        """
        if self.n_nodes == 0:
            return [], 0.0, 0.0

        start = time.time()
        T = self.params.max_iterations

        # Initialise population in continuous space
        X = np.random.uniform(-1.0, 1.0, (self.params.pop_size, self.n_nodes))
        routes_pop = [self._decode(x) for x in X]
        profits = [self._evaluate(r) for r in routes_pop]

        best_idx = int(np.argmax(profits))
        X_best = X[best_idx].copy()
        best_routes = copy.deepcopy(routes_pop[best_idx])
        best_profit = profits[best_idx]
        best_cost = self._cost(best_routes)

        for t in range(T):
            if time.time() - start > self.params.time_limit:
                break

            # Control parameter decays from a_max → 0
            a = self.params.a_max * (1.0 - t / T)

            for i in range(self.params.pop_size):
                r1 = random.uniform(0, a)
                r2 = random.uniform(0, 2 * math.pi)
                r3 = random.uniform(0, 2)
                r4 = random.random()

                diff = r3 * X_best - X[i]

                if r4 < 0.5:
                    X[i] = X[i] + r1 * math.sin(r2) * np.abs(diff)
                else:
                    X[i] = X[i] + r1 * math.cos(r2) * np.abs(diff)

                # Decode and evaluate
                routes_pop[i] = self._decode(X[i])
                profits[i] = self._evaluate(routes_pop[i])

                if profits[i] > best_profit:
                    X_best = X[i].copy()
                    best_routes = copy.deepcopy(routes_pop[i])
                    best_profit = profits[i]
                    best_cost = self._cost(best_routes)

            self._viz_record(
                iteration=t,
                best_profit=best_profit,
                best_cost=best_cost,
                a=a,
            )

        return best_routes, best_profit, best_cost

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _decode(self, x: np.ndarray) -> List[List[int]]:
        """
        Decode a continuous position vector to a discrete routing solution.

        Steps:
          1. Apply sigmoid binarisation: b_j = 1 if sigmoid(x_j) > 0.5.
          2. Among selected nodes (b_j=1), order by Largest Rank Value (LRV)
             — i.e., sort by x_j descending to produce visit sequence.
          3. Insert mandatory nodes that were not selected.
          4. Build routes via greedy_insertion.

        Args:
            x: Continuous position vector of length n_nodes.

        Returns:
            Routing solution as list of routes.
        """
        sigmoid = 1.0 / (1.0 + np.exp(-x))

        mandatory_set = set(self.mandatory_nodes)
        selected_nodes: List[int] = []

        # Always include mandatory nodes
        for node in self.nodes:
            if node in mandatory_set:
                selected_nodes.append(node)

        # Include optional nodes where sigmoid > 0.5
        total_load = sum(self.wastes.get(n, 0.0) for n in selected_nodes)
        optional_sorted = sorted(
            [(sigmoid[idx], self.nodes[idx]) for idx in range(self.n_nodes) if self.nodes[idx] not in mandatory_set],
            reverse=True,
        )

        for _, node in optional_sorted:
            w = self.wastes.get(node, 0.0)
            if sigmoid[self.nodes.index(node)] > 0.5 and total_load + w <= self.capacity:
                selected_nodes.append(node)
                total_load += w

        if not selected_nodes:
            return []

        from logic.src.policies.operators.repair.greedy import greedy_insertion

        routes = greedy_insertion(
            [],
            selected_nodes,
            self.dist_matrix,
            self.wastes,
            self.capacity,
            R=self.R,
            mandatory_nodes=self.mandatory_nodes,
            expand_pool=False,
        )
        return routes

    def _evaluate(self, routes: List[List[int]]) -> float:
        """Net profit for a set of routes."""
        if not routes:
            return 0.0
        rev = sum(self.wastes.get(n, 0.0) * self.R for r in routes for n in r)
        return rev - self._cost(routes) * self.C

    def _cost(self, routes: List[List[int]]) -> float:
        """Total routing distance."""
        total = 0.0
        for route in routes:
            if not route:
                continue
            total += self.dist_matrix[0][route[0]]
            for k in range(len(route) - 1):
                total += self.dist_matrix[route[k]][route[k + 1]]
            total += self.dist_matrix[route[-1]][0]
        return total
