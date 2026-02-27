"""
League Championship Algorithm (LCA) for VRPP.

Teams play weekly matches in a multi-round-robin schedule.  The loser
analyses the winner's routing structure and generates a new "formation"
(updated route set) for the following week.  A diversity-preserving
infeasibility tolerance allows mildly infeasible solutions — those whose
profit exceeds that of strictly feasible competitors by less than
`tolerance_pct` — to rank above feasible but low-quality solutions.

This controlled infeasibility provides topological bridging between isolated
feasible basins, a property that proves particularly valuable in highly
constrained VRPP variants such as helicopter offshore routing.

Reference:
    Survey §"League Championship Algorithm" — helicopter offshore routing.
"""

import contextlib
import copy
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.tracking.viz_mixin import PolicyVizMixin

from ..operators.destroy_operators import worst_removal
from ..operators.repair_operators import greedy_insertion
from .params import LCAParams


class LCASolver(PolicyVizMixin):
    """
    League Championship Algorithm solver for VRPP.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: LCAParams,
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
        self.mandatory_set = set(self.mandatory_nodes)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def solve(self) -> Tuple[List[List[int]], float, float]:
        """
        Run LCA and return the best feasible solution.

        Returns:
            Tuple of (routes, profit, cost).
        """
        if self.n_nodes == 0:
            return [], 0.0, 0.0

        start = time.time()

        # Initialise teams (routing solutions)
        teams: List[List[List[int]]] = [self._random_solution() for _ in range(self.params.n_teams)]
        profits: List[float] = [self._evaluate(t) for t in teams]

        best_idx = int(np.argmax(profits))
        best_routes = copy.deepcopy(teams[best_idx])
        best_profit = profits[best_idx]
        best_cost = self._cost(best_routes)

        for iteration in range(self.params.max_iterations):
            if time.time() - start > self.params.time_limit:
                break

            # Random round-robin schedule for this week
            order = list(range(self.params.n_teams))
            random.shuffle(order)

            for k in range(0, len(order) - 1, 2):
                a_idx = order[k]
                b_idx = order[k + 1]

                pa = profits[a_idx]
                pb = profits[b_idx]

                # Determine winner/loser with infeasibility tolerance
                # A team within `tolerance_pct` of its opponent is allowed
                # to win based on diversity (random coin flip)
                delta = abs(pa - pb)
                tolerance = self.params.tolerance_pct * (abs(pa) + abs(pb) + 1e-9) / 2.0

                if delta <= tolerance:
                    # Close match — random winner
                    winner, loser = (a_idx, b_idx) if random.random() < 0.5 else (b_idx, a_idx)
                elif pa >= pb:
                    winner, loser = a_idx, b_idx
                else:
                    winner, loser = b_idx, a_idx

                # Loser generates new formation
                if random.random() < self.params.crossover_prob:
                    new_team = self._crossover(teams[loser], teams[winner])
                else:
                    new_team = self._perturb(teams[loser])

                new_profit = self._evaluate(new_team)

                # Accept new formation (LCA always updates the loser)
                teams[loser] = new_team
                profits[loser] = new_profit

                # Track best feasible solution
                if new_profit > best_profit:
                    best_routes = copy.deepcopy(new_team)
                    best_profit = new_profit
                    best_cost = self._cost(best_routes)

            self._viz_record(
                iteration=iteration,
                best_profit=best_profit,
                best_cost=best_cost,
                n_teams=self.params.n_teams,
            )

        return best_routes, best_profit, best_cost

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _random_solution(self) -> List[List[int]]:
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

    def _perturb(self, routes: List[List[int]]) -> List[List[int]]:
        """
        Formation perturbation: worst removal + greedy re-insertion.

        Args:
            routes: Current team routes.

        Returns:
            Perturbed routes.
        """
        n = max(3, self.params.n_removal)
        try:
            partial, removed = worst_removal(routes, n, self.dist_matrix)
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

    def _crossover(self, loser_routes: List[List[int]], winner_routes: List[List[int]]) -> List[List[int]]:
        """
        Generate a new formation by injecting a segment from the winner's routes.

        The loser adopts a contiguous segment of nodes from the winner's flat
        tour that it does not already service.  The resulting child is then
        repaired via greedy_insertion to maintain feasibility.

        Args:
            loser_routes: Losing team's routes.
            winner_routes: Winning team's routes.

        Returns:
            Child routing solution.
        """
        winner_flat = [n for r in winner_routes for n in r]
        loser_visited = {n for r in loser_routes for n in r}

        if len(winner_flat) < 2:
            return self._perturb(loser_routes)

        seg_len = max(1, len(winner_flat) // 4)
        start_idx = random.randint(0, max(0, len(winner_flat) - seg_len))
        segment = winner_flat[start_idx : start_idx + seg_len]
        new_nodes = [n for n in segment if n not in loser_visited]

        child = copy.deepcopy(loser_routes)

        # First remove some worst nodes to make room
        n_remove = min(len(new_nodes), max(1, self.params.n_removal))
        with contextlib.suppress(Exception):
            child, _ = worst_removal(child, n_remove, self.dist_matrix)

        if new_nodes:
            with contextlib.suppress(Exception):
                child = greedy_insertion(
                    child,
                    new_nodes,
                    self.dist_matrix,
                    self.wastes,
                    self.capacity,
                    R=self.R,
                    mandatory_nodes=self.mandatory_nodes,
                )

        # Apply comprehensive local search
        from logic.src.policies.local_search.local_search_aco import ACOLocalSearch

        ls = ACOLocalSearch(self.dist_matrix, self.wastes, self.capacity, self.R, self.C, self.params)
        return ls.optimize(child)

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
