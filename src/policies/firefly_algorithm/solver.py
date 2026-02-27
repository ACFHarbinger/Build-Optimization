"""
Discrete Firefly Algorithm (DFA) for VRPP.

Adapts the continuous FA to discrete routing by replacing Euclidean distance
with swap distance (number of non-matching edges) and replacing real-valued
position updates with node-extraction / guided-insertion movements governed
by a three-term favourability score.

Reference:
    Survey §"Firefly Algorithm" — Rich-VRP newspaper distribution systems.
"""

import contextlib
import copy
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.tracking.viz_mixin import PolicyVizMixin

from ..operators.destroy_operators import random_removal
from ..operators.repair_operators import greedy_insertion
from .params import FAParams


class FASolver(PolicyVizMixin):
    """
    Discrete Firefly Algorithm solver for VRPP.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: FAParams,
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
        Run the Discrete Firefly Algorithm.

        Returns:
            Tuple of (routes, profit, cost).
        """
        if self.n_nodes == 0:
            return [], 0.0, 0.0

        start = time.time()

        population = [self._new_firefly() for _ in range(self.params.pop_size)]
        profits = [self._evaluate(f) for f in population]

        best_idx = int(np.argmax(profits))
        best_routes = copy.deepcopy(population[best_idx])
        best_profit = profits[best_idx]
        best_cost = self._cost(best_routes)

        for iteration in range(self.params.max_iterations):
            if time.time() - start > self.params.time_limit:
                break

            # Pairwise attraction: dimmer firefly moves toward brighter one
            for i in range(self.params.pop_size):
                moved = False
                for j in range(self.params.pop_size):
                    if profits[j] <= profits[i]:
                        continue

                    d = self._swap_distance(population[i], population[j])
                    beta = self.params.beta0 * np.exp(-self.params.gamma * d * d)

                    if random.random() < beta:
                        new_routes = self._attract(population[i], population[j])
                        new_profit = self._evaluate(new_routes)
                        if new_profit > profits[i]:
                            population[i] = new_routes
                            profits[i] = new_profit
                            moved = True

                # Random walk if not attracted or by chance
                if not moved or random.random() < self.params.alpha_rnd:
                    rw = self._random_walk(population[i])
                    rw_profit = self._evaluate(rw)
                    if rw_profit > profits[i]:
                        population[i] = rw
                        profits[i] = rw_profit

                # Update global best
                if profits[i] > best_profit:
                    best_routes = copy.deepcopy(population[i])
                    best_profit = profits[i]
                    best_cost = self._cost(best_routes)

            self._viz_record(
                iteration=iteration,
                best_profit=best_profit,
                best_cost=best_cost,
                population_size=self.params.pop_size,
            )

        return best_routes, best_profit, best_cost

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _new_firefly(self) -> List[List[int]]:
        """Generate a random feasible routing solution."""
        return self._build_random_solution()

    def _build_random_solution(self) -> List[List[int]]:
        """Order-dependent sequential construction (matches ALNS style).

        Random node ordering causes different capacity cutoffs, creating
        genuinely diverse initial solutions. Uses self.C for the profitability
        check so that economics are consistent with the solver's _evaluate().
        """
        from logic.src.policies.operators.heuristics.initialization import build_nn_routes

        optimized_routes = build_nn_routes(
            nodes=self.nodes,
            mandatory_nodes=self.mandatory_nodes,
            wastes=self.wastes,
            capacity=self.capacity,
            dist_matrix=self.dist_matrix,
            R=self.R,
            C=self.C,
        )
        return optimized_routes

    def _swap_distance(self, routes_a: List[List[int]], routes_b: List[List[int]]) -> int:
        """
        Compute discrete swap distance between two routing solutions.

        Defined as the number of edges present in one solution but not the
        other (symmetric edge set difference).

        Args:
            routes_a: First route set.
            routes_b: Second route set.

        Returns:
            Integer swap distance ≥ 0.
        """
        edges_a = self._edge_set(routes_a)
        edges_b = self._edge_set(routes_b)
        return len(edges_a.symmetric_difference(edges_b))

    @staticmethod
    def _edge_set(routes: List[List[int]]) -> set:
        """Extract the set of directed edges (including depot arcs) from routes."""
        edges = set()
        for route in routes:
            if not route:
                continue
            edges.add((0, route[0]))
            for k in range(len(route) - 1):
                edges.add((route[k], route[k + 1]))
            edges.add((route[-1], 0))
        return edges

    def _attract(self, dim_routes: List[List[int]], bright_routes: List[List[int]]) -> List[List[int]]:
        """
        Move dim firefly toward bright firefly via guided node insertion.

        Extracts nodes that are in the bright solution but not in the dim
        solution, scores each by favourability, and inserts them greedily.

        Favourability score per candidate node n:
            score(n) = α_p * profit(n) + β_w * willingness(n) - γ_c * insertion_cost(n)

        Args:
            dim_routes: Routes of the less bright firefly.
            bright_routes: Routes of the brighter firefly.

        Returns:
            Updated routes for the dim firefly.
        """
        dim_visited = {n for r in dim_routes for n in r}
        bright_visited = {n for r in bright_routes for n in r}
        candidates = [n for n in bright_visited if n not in dim_visited]

        if not candidates:
            return copy.deepcopy(dim_routes)

        # Score candidates
        scored = []
        for node in candidates:
            profit_n = self.wastes.get(node, 0.0) * self.R
            willingness = self.wastes.get(node, 0.0)  # fill level as willingness proxy
            ins_cost = self._best_insertion_cost(node, dim_routes)
            score = (
                self.params.alpha_profit * profit_n
                + self.params.beta_will * willingness
                - self.params.gamma_cost * ins_cost
            )
            scored.append((score, node))

        scored.sort(reverse=True)
        selected = [node for _, node in scored]

        routes = copy.deepcopy(dim_routes)
        with contextlib.suppress(Exception):
            routes = greedy_insertion(
                routes,
                selected,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                R=self.R,
                mandatory_nodes=self.mandatory_nodes,
            )
            # Apply comprehensive local search
            from logic.src.policies.local_search.local_search_aco import ACOLocalSearch

            ls = ACOLocalSearch(self.dist_matrix, self.wastes, self.capacity, self.R, self.C, self.params)
            return ls.optimize(routes)
        return routes

    def _best_insertion_cost(self, node: int, routes: List[List[int]]) -> float:
        """
        Compute the minimum insertion cost of a node into existing routes.

        Args:
            node: Node index to insert.
            routes: Existing routes.

        Returns:
            Minimum additional distance required to insert the node.
        """
        min_cost = float("inf")
        for route in routes:
            for i in range(len(route) + 1):
                prev = 0 if i == 0 else route[i - 1]
                nxt = 0 if i == len(route) else route[i]
                c = self.dist_matrix[prev][node] + self.dist_matrix[node][nxt] - self.dist_matrix[prev][nxt]
                if c < min_cost:
                    min_cost = c
        # If no routes, cost = depot → node → depot
        if not routes:
            min_cost = self.dist_matrix[0][node] + self.dist_matrix[node][0]
        return max(0.0, min_cost)

    def _random_walk(self, routes: List[List[int]]) -> List[List[int]]:
        """
        Random walk: remove one node randomly and reinsert greedily.

        Args:
            routes: Current routes.

        Returns:
            Perturbed routes.
        """
        try:
            n_rem = max(3, self.params.n_removal)
            partial, removed = random_removal(routes, n_rem)
            repaired = greedy_insertion(
                partial,
                removed,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                R=self.R,
                mandatory_nodes=self.mandatory_nodes,
            )
            # Apply comprehensive local search
            from logic.src.policies.local_search.local_search_aco import ACOLocalSearch

            ls = ACOLocalSearch(self.dist_matrix, self.wastes, self.capacity, self.R, self.C, self.params)
            return ls.optimize(repaired)
        except Exception:
            return copy.deepcopy(routes)

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
