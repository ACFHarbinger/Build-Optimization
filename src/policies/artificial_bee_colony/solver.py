"""
Artificial Bee Colony (ABC) algorithm for VRPP.

Three agent types — employed, onlooker, and scout bees — cooperate to
explore and exploit the routing solution space without requiring gradient
information, making ABC naturally suited to the discontinuous profit
landscapes of the VRPP.

Reference:
    Survey §"Artificial Bee Colony" — real-world VRP with time windows.
"""

import copy
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.tracking.viz_mixin import PolicyVizMixin

from ..operators import (
    greedy_insertion,
    random_removal,
)
from .params import ABCParams


class ABCSolver(PolicyVizMixin):
    """
    Artificial Bee Colony solver for VRPP.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: ABCParams,
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
        Run ABC and return the best solution found.

        Returns:
            Tuple of (routes, profit, cost).
        """
        if self.n_nodes == 0:
            return [], 0.0, 0.0

        start = time.time()

        # Initialise food sources (employed bees)
        sources = [self._new_source() for _ in range(self.params.n_sources)]
        profits = [self._evaluate(s) for s in sources]
        trials = [0] * self.params.n_sources

        best_idx = int(np.argmax(profits))
        best_routes = copy.deepcopy(sources[best_idx])
        best_profit = profits[best_idx]
        best_cost = self._cost(best_routes)

        for iteration in range(self.params.max_iterations):
            if time.time() - start > self.params.time_limit:
                break

            # --- Employed bee phase ---
            for i in range(self.params.n_sources):
                # Select a random peer to guide the interpolation
                peer_idx = random.choice([x for x in range(self.params.n_sources) if x != i])
                neighbour = self._perturb(sources[i], sources[peer_idx])
                nb_profit = self._evaluate(neighbour)

                if nb_profit > profits[i]:
                    sources[i] = neighbour
                    profits[i] = nb_profit
                    trials[i] = 0
                else:
                    trials[i] += 1

            # --- Onlooker bee phase ---
            # Fitness for roulette wheel (shift to keep positive)
            min_p = min(profits)
            shifted = [p - min_p + 1e-9 for p in profits]
            total = sum(shifted)
            probs = [s / total for s in shifted]

            for _ in range(self.params.n_sources):
                i = self._roulette(probs)
                peer_idx = random.choice([x for x in range(self.params.n_sources) if x != i])
                neighbour = self._perturb(sources[i], sources[peer_idx])
                nb_profit = self._evaluate(neighbour)

                if nb_profit > profits[i]:
                    sources[i] = neighbour
                    profits[i] = nb_profit
                    trials[i] = 0
                else:
                    trials[i] += 1

            # Update global best
            for i in range(self.params.n_sources):
                if profits[i] > best_profit:
                    best_routes = copy.deepcopy(sources[i])
                    best_profit = profits[i]
                    best_cost = self._cost(best_routes)

            # --- Scout bee phase ---
            for i in range(self.params.n_sources):
                if trials[i] > self.params.limit:
                    sources[i] = self._new_source()
                    profits[i] = self._evaluate(sources[i])
                    trials[i] = 0

            self._viz_record(
                iteration=iteration,
                best_profit=best_profit,
                best_cost=best_cost,
                n_sources=self.params.n_sources,
            )

        return best_routes, best_profit, best_cost

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _new_source(self) -> List[List[int]]:
        """Generate a new random feasible food source."""
        return self._build_random_solution()

    def _build_random_solution(self) -> List[List[int]]:
        """
        Builds a random initial solution applying nearest neighbor ordering.
        """
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

    def _perturb(self, current: List[List[int]], peer: List[List[int]]) -> List[List[int]]:
        """
        Cross-solution interpolation: extracts nodes from a peer and injects
        them into the current solution, mimicking the v_ij = x_ij + φ(x_ij - x_kj) equation.
        """
        if not current or not peer:
            return copy.deepcopy(current)

        n = max(3, self.params.n_removal)

        peer_nodes = [node for route in peer for node in route]
        if not peer_nodes:
            return copy.deepcopy(current)

        # Take a random subset of nodes from peer
        selected_peer_nodes = random.sample(peer_nodes, min(n, len(peer_nodes)))

        current_copy = copy.deepcopy(current)
        # Remove these nodes from current
        for route in current_copy:
            for node in selected_peer_nodes:
                if node in route:
                    route.remove(node)

        current_copy = [r for r in current_copy if r]

        # Also randomly remove some additional nodes for diversity
        try:
            partial, additional_removed = random_removal(current_copy, n)
            to_insert = list(set(selected_peer_nodes + additional_removed))

            repaired = greedy_insertion(
                partial,
                to_insert,
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
            return copy.deepcopy(current)

    @staticmethod
    def _roulette(probs: List[float]) -> int:
        """
        Roulette-wheel selection.
        """
        r = random.random()
        cumulative = 0.0
        for i, p in enumerate(probs):
            cumulative += p
            if r <= cumulative:
                return i
        return len(probs) - 1

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
