"""
Quantum-Inspired Differential Evolution (QDE) for VRPP.

Represents each candidate as a quantum amplitude vector q ∈ [0,1]^N.
Standard DE mutation / crossover operates in this continuous space; the
trial vector is collapsed to a discrete routing solution by ranking node
amplitudes and calling greedy_insertion.

Reference:
    Survey §"Differential Evolution" — quantum-inspired representation for discrete OP.
"""

import copy
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.tracking.viz_mixin import PolicyVizMixin

from .params import QDEParams


class QDESolver(PolicyVizMixin):
    """
    Quantum-Inspired Differential Evolution solver for VRPP.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: QDEParams,
        mandatory_nodes: Optional[List[int]] = None,
    ):
        self.dist_matrix = dist_matrix
        self.wastes = wastes
        self.capacity = capacity
        self.R = R
        self.C = C
        self.params = params
        self.mandatory_nodes = mandatory_nodes or []
        self.n_nodes = len(dist_matrix) - 1  # Exclude depot
        self.nodes = list(range(1, self.n_nodes + 1))

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def solve(self) -> Tuple[List[List[int]], float, float]:
        """
        Run QDE and return the best solution found.

        Returns:
            Tuple of (routes, profit, cost).
        """
        if self.n_nodes == 0:
            return [], 0.0, 0.0

        start = time.time()
        pop_size = self.params.pop_size

        # Initialise population: amplitude vectors ∈ [0,1]^N
        population = [np.random.uniform(0.0, 1.0, self.n_nodes) for _ in range(pop_size)]
        routes_pop = [self._collapse(amp) for amp in population]
        profits = [self._evaluate(r) for r in routes_pop]

        best_idx = int(np.argmax(profits))
        best_routes = copy.deepcopy(routes_pop[best_idx])
        best_profit = profits[best_idx]
        best_cost = self._cost(best_routes)

        for iteration in range(self.params.max_iterations):
            if time.time() - start > self.params.time_limit:
                break

            for i in range(pop_size):
                # --- Mutation ---
                candidates = [j for j in range(pop_size) if j != i]
                r1, r2, r3 = random.sample(candidates, 3)
                mutant = np.clip(
                    population[r1] + self.params.F * (population[r2] - population[r3]),
                    0.0,
                    1.0,
                )

                # --- Crossover (binomial) ---
                j_rand = random.randint(0, self.n_nodes - 1)
                trial = np.where(
                    (np.random.uniform(0.0, 1.0, self.n_nodes) < self.params.CR) | (np.arange(self.n_nodes) == j_rand),
                    mutant,
                    population[i],
                )

                # --- Collapse → discrete routes ---
                trial_routes = self._collapse(trial)
                trial_profit = self._evaluate(trial_routes)

                # --- Greedy selection ---
                if trial_profit >= profits[i]:
                    population[i] = trial
                    routes_pop[i] = trial_routes
                    profits[i] = trial_profit

                    if trial_profit > best_profit:
                        best_routes = copy.deepcopy(trial_routes)
                        best_profit = trial_profit
                        best_cost = self._cost(best_routes)

            self._viz_record(
                iteration=iteration,
                best_profit=best_profit,
                best_cost=best_cost,
                population_size=pop_size,
            )

        return best_routes, best_profit, best_cost

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _collapse(self, amplitudes: np.ndarray) -> List[List[int]]:
        """
        Collapse amplitude vector to a discrete routing solution.

        Nodes are ranked by amplitude (descending).  Mandatory nodes are
        always included.  Optional nodes are added in amplitude order until
        capacity would be exceeded, at which point they are skipped.

        Args:
            amplitudes: Amplitude vector of length n_nodes.

        Returns:
            Routes built by greedy_insertion.
        """
        ranked = sorted(range(self.n_nodes), key=lambda j: amplitudes[j], reverse=True)

        selected: List[int] = []
        mandatory_set = set(self.mandatory_nodes)
        total_load = 0.0

        for j in ranked:
            node = j + 1  # 1-based index (depot is 0)
            waste = self.wastes.get(node, 0.0)
            if node in mandatory_set or total_load + waste <= self.capacity:
                selected.append(node)
                total_load += waste

        if not selected:
            return []

        from logic.src.policies.operators.repair.greedy import greedy_insertion

        routes = greedy_insertion(
            [],
            selected,
            self.dist_matrix,
            self.wastes,
            self.capacity,
            R=self.R,
            mandatory_nodes=self.mandatory_nodes,
            expand_pool=False,
        )
        return routes

    def _evaluate(self, routes: List[List[int]]) -> float:
        """
        Compute net profit for a set of routes.

        Args:
            routes: List of routes (each a list of node indices).

        Returns:
            Net profit (revenue − travel cost).
        """
        if not routes:
            return 0.0
        rev = sum(self.wastes.get(n, 0.0) * self.R for r in routes for n in r)
        cost = self._cost(routes)
        return rev - cost * self.C

    def _cost(self, routes: List[List[int]]) -> float:
        """
        Compute total routing distance.

        Args:
            routes: List of routes.

        Returns:
            Total distance.
        """
        total = 0.0
        for route in routes:
            if not route:
                continue
            total += self.dist_matrix[0][route[0]]
            for k in range(len(route) - 1):
                total += self.dist_matrix[route[k]][route[k + 1]]
            total += self.dist_matrix[route[-1]][0]
        return total
