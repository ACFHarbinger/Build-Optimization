"""
Harmony Search (HS) algorithm for VRPP.

Models the optimisation process as a musical improvisation session.  The
Harmony Memory stores the most profitable route configurations found so
far.  A new harmony (routing solution) is built node-by-node by consulting
the HM (HMCR), applying pitch adjustment (PAR), or selecting a random
unvisited node.

Near-zero benchmark errors (<0.01%) on Orienteering Problem instances have
been reported in the literature when HMCR and PAR are carefully tuned.

Reference:
    Survey §"Harmony Search" — Orienteering Problem, <0.01% average error.
"""

import copy
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.tracking.viz_mixin import PolicyVizMixin

from ..operators.repair_operators import greedy_insertion
from .params import HSParams


class HSSolver(PolicyVizMixin):
    """
    Harmony Search solver for VRPP.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: HSParams,
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
        Run Harmony Search and return the best routing solution.

        Returns:
            Tuple of (routes, profit, cost).
        """
        if self.n_nodes == 0:
            return [], 0.0, 0.0

        start = time.time()

        # Initialise Harmony Memory
        hm: List[List[List[int]]] = [self._random_harmony() for _ in range(self.params.hm_size)]
        hm_profits = [self._evaluate(h) for h in hm]

        best_idx = int(np.argmax(hm_profits))
        best_routes = copy.deepcopy(hm[best_idx])
        best_profit = hm_profits[best_idx]
        best_cost = self._cost(best_routes)

        for iteration in range(self.params.max_iterations):
            if time.time() - start > self.params.time_limit:
                break

            # Improvise a new harmony (routing solution)
            new_harmony = self._improvise(hm)
            new_profit = self._evaluate(new_harmony)

            # Update HM: replace worst if new harmony is better
            worst_idx = int(np.argmin(hm_profits))
            if new_profit > hm_profits[worst_idx]:
                hm[worst_idx] = new_harmony
                hm_profits[worst_idx] = new_profit

                if new_profit > best_profit:
                    best_routes = copy.deepcopy(new_harmony)
                    best_profit = new_profit
                    best_cost = self._cost(best_routes)

            self._viz_record(
                iteration=iteration,
                best_profit=best_profit,
                best_cost=best_cost,
                hm_size=self.params.hm_size,
            )

        return best_routes, best_profit, best_cost

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _random_harmony(self) -> List[List[int]]:
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

    def _improvise(self, hm: List[List[List[int]]]) -> List[List[int]]:
        """
        Improvise a new harmony using HMCR, PAR, and random selection.

        Builds a candidate node sequence, then routes it via greedy_insertion.

        Args:
            hm: Current Harmony Memory.

        Returns:
            Newly improvised routing solution.
        """
        set(self.mandatory_nodes)
        candidate_nodes: List[int] = []

        # Pool all nodes that appear in any HM solution
        hm_node_pool: List[List[int]] = []
        for harmony in hm:
            flat = [n for r in harmony for n in r]
            hm_node_pool.append(flat)

        unvisited = set(self.nodes)

        for _node in self.nodes:
            if not unvisited:
                break

            if random.random() < self.params.HMCR:
                # Select from HM: pick a random harmony, take its node at this slot
                src_flat = random.choice(hm_node_pool)
                if src_flat:
                    hm_node = random.choice(src_flat)
                    selected = hm_node if hm_node in unvisited else random.choice(list(unvisited))
                else:
                    selected = random.choice(list(unvisited))

                # Pitch adjustment: swap with a random neighbour
                if random.random() < self.params.PAR:
                    neighbours = self._nearest_unvisited(selected, unvisited - {selected})
                    if neighbours:
                        selected = neighbours[0]
            else:
                # Random selection
                selected = random.choice(list(unvisited))

            candidate_nodes.append(selected)
            unvisited.discard(selected)

        # Add any mandatory nodes not yet in candidate list
        for mn in self.mandatory_nodes:
            if mn not in candidate_nodes:
                candidate_nodes.append(mn)

        if not candidate_nodes:
            return []

        try:
            routes = greedy_insertion(
                [],
                candidate_nodes,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                R=self.R,
                mandatory_nodes=self.mandatory_nodes,
            )
            from logic.src.policies.local_search.local_search_aco import ACOLocalSearch

            ls = ACOLocalSearch(self.dist_matrix, self.wastes, self.capacity, self.R, self.C, self.params)
            routes = ls.optimize(routes)
        except Exception:
            routes = []
        return routes

    def _nearest_unvisited(self, node: int, unvisited: set) -> List[int]:
        """
        Return unvisited nodes sorted by distance to the given node.

        Args:
            node: Reference node.
            unvisited: Set of unvisited node indices.

        Returns:
            Sorted list of unvisited nodes (nearest first).
        """
        if not unvisited:
            return []
        return sorted(list(unvisited), key=lambda n: self.dist_matrix[node][n])

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
