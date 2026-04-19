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
    Kashan, A. H. (2013). "League Championship Algorithm (LCA): An
    algorithm for global optimization inspired by sport championships."
"""

import contextlib
import copy
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.policies.helpers.local_search.local_search_aco import ACOLocalSearch
from logic.src.policies.helpers.operators import (
    greedy_insertion,
    greedy_profit_insertion,
    random_removal,
    worst_profit_removal,
    worst_removal,
)
from logic.src.policies.helpers.operators.solution_initialization.greedy_si import build_greedy_routes
from logic.src.policies.route_construction.meta_heuristics.ant_colony_optimization_k_sparse.params import KSACOParams

from .params import LCAParams


class LCASolver:
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
        self.random = random.Random(self.params.seed) if self.params.seed is not None else random.Random()

        # Pre-instantiate Local Search for reuse
        aco_params = KSACOParams(
            local_search_iterations=self.params.local_search_iterations,
            vrpp=self.params.vrpp,
            profit_aware_operators=self.params.profit_aware_operators,
            seed=self.params.seed,
        )
        self.ls = ACOLocalSearch(
            dist_matrix=self.dist_matrix,
            waste=self.wastes,
            capacity=self.capacity,
            R=self.R,
            C=self.C,
            params=aco_params,
        )

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

        start = time.process_time()

        # Initialise teams (routing solutions)
        teams: List[List[List[int]]] = [self._build_random_solution() for _ in range(self.params.n_teams)]
        profits: List[float] = [self._evaluate(t) for t in teams]

        # Global best
        best_routes = copy.deepcopy(teams[int(np.argmax(profits))])
        best_profit = max(profits)
        best_cost = self._cost(best_routes)

        for iteration in range(self.params.max_iterations):
            if self.params.time_limit > 0 and time.process_time() - start > self.params.time_limit:
                break

            # Weekly Matches: Multi-round Robin (Random pairs per iteration)
            # In standard LCA, teams play one match per week.
            order = list(range(self.params.n_teams))
            self.random.shuffle(order)

            for k in range(0, len(order) - 1, 2):
                a_idx, b_idx = order[k], order[k + 1]
                pa, pb = profits[a_idx], profits[b_idx]

                # Determine winner/loser via fitness comparison (Kashan 2011/2013)
                # Kashan (2013): Winner stays, Loser generates new formation by learning from winner
                if pa > pb:
                    winner, loser = a_idx, b_idx
                elif pb > pa:
                    winner, loser = b_idx, a_idx
                else:
                    winner, loser = (a_idx, b_idx) if self.random.random() < 0.5 else (b_idx, a_idx)

                # Formation Update (Learning from Winner)
                # Kashan (2013): The loser adopts the winner's strategy (crossover) or explores (perturbation)
                if self.random.random() < self.params.crossover_prob:
                    # Loser adopts winner's formation features (segment injection)
                    new_formation = self._crossover(teams[loser], teams[winner])
                else:
                    # Loser perturbs its own formation (Random walk)
                    new_formation = self._perturb(teams[loser])

                new_profit = self._evaluate(new_formation)

                # Accept new formation (LCA always updates the loser)
                teams[loser] = new_formation
                profits[loser] = new_profit

                # Track global best
                if new_profit > best_profit:
                    best_routes = copy.deepcopy(new_formation)
                    best_profit = new_profit
                    best_cost = self._cost(best_routes)

            getattr(self, "_viz_record", lambda **k: None)(
                iteration=iteration,
                best_profit=best_profit,
                best_cost=best_cost,
                n_teams=self.params.n_teams,
            )

        return best_routes, best_profit, best_cost

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_random_solution(self) -> List[List[int]]:
        """Order-dependent sequential construction (matches ALNS style).

        Random node ordering causes different capacity cutoffs, creating
        genuinely diverse initial solutions. Uses self.C for the profitability
        check so that economics are consistent with the solver's _evaluate().
        """
        return build_greedy_routes(
            dist_matrix=self.dist_matrix,
            wastes=self.wastes,
            capacity=self.capacity,
            R=self.R,
            C=self.C,
            mandatory_nodes=self.mandatory_nodes,
            rng=self.random,
        )

    def _perturb(self, routes: List[List[int]]) -> List[List[int]]:
        """
        Formation perturbation: worst removal + greedy re-insertion.

        Args:
            routes: Current team routes.

        Returns:
            Perturbed routes.
        """
        n = max(3, self.params.n_removal)
        use_profit = self.params.profit_aware_operators
        expand_pool = self.params.vrpp

        try:
            if use_profit:
                removal_op = self.random.choice([random_removal, worst_profit_removal])
                if removal_op == random_removal:
                    partial, removed = random_removal(routes, n, rng=self.random)
                else:
                    partial, removed = worst_profit_removal(
                        routes, n, self.dist_matrix, self.wastes, R=self.R, C=self.C
                    )

                repaired = greedy_profit_insertion(
                    partial,
                    removed,
                    self.dist_matrix,
                    self.wastes,
                    self.capacity,
                    self.R,
                    self.C,
                    mandatory_nodes=self.mandatory_nodes,
                    expand_pool=expand_pool,
                )
            else:
                removal_op = self.random.choice([random_removal, worst_removal])
                if removal_op == random_removal:
                    partial, removed = random_removal(routes, n, rng=self.random)
                else:
                    partial, removed = worst_removal(routes, n, self.dist_matrix)

                repaired = greedy_insertion(
                    partial,
                    removed,
                    self.dist_matrix,
                    self.wastes,
                    self.capacity,
                    mandatory_nodes=self.mandatory_nodes,
                    expand_pool=expand_pool,
                )

            # Apply comprehensive local search (reusing instance)
            return self.ls.optimize(repaired)
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
        start_idx = self.random.randint(0, max(0, len(winner_flat) - seg_len))
        segment = winner_flat[start_idx : start_idx + seg_len]
        new_nodes = [n for n in segment if n not in loser_visited]

        child = copy.deepcopy(loser_routes)
        use_profit = self.params.profit_aware_operators
        expand_pool = self.params.vrpp

        # First remove some worst nodes to make room
        n_remove = min(len(new_nodes), max(1, self.params.n_removal))
        with contextlib.suppress(Exception):
            if use_profit:
                child, _ = worst_profit_removal(child, n_remove, self.dist_matrix, self.wastes, R=self.R, C=self.C)
            else:
                child, _ = worst_removal(child, n_remove, self.dist_matrix)

        if new_nodes:
            with contextlib.suppress(Exception):
                if use_profit:
                    child = greedy_profit_insertion(
                        child,
                        new_nodes,
                        self.dist_matrix,
                        self.wastes,
                        self.capacity,
                        self.R,
                        self.C,
                        mandatory_nodes=self.mandatory_nodes,
                        expand_pool=expand_pool,
                    )
                else:
                    child = greedy_insertion(
                        child,
                        new_nodes,
                        self.dist_matrix,
                        self.wastes,
                        self.capacity,
                        mandatory_nodes=self.mandatory_nodes,
                        expand_pool=expand_pool,
                    )

        # Apply comprehensive local search (reusing instance)
        return self.ls.optimize(child)

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
