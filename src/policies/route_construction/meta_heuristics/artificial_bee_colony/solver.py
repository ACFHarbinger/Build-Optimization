"""
Artificial Bee Colony (ABC) algorithm for VRPP.

Three agent types — employed, onlooker, and scout bees — cooperate to
explore and exploit the routing solution space without requiring gradient
information, making ABC naturally suited to the discontinuous profit
landscapes of the VRPP.

Reference:
    Rechenberg, I. (1973). "Evolutionsstrategie: Optimierung technischer
    Systeme nach Prinzipien der biologischen Evolution."
    Yao, B., Yan, Q., Zhang, M., & Yang, Y. "Improved artificial bee
    colony algorithm for vehicle routing problem with time windows", 2017.
"""

import copy
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.policies.helpers.local_search.local_search_aco import (
    ACOLocalSearch,
)
from logic.src.policies.helpers.operators import (
    greedy_insertion,
    greedy_profit_insertion,
    random_removal,
    regret_2_insertion,
    regret_2_profit_insertion,
    worst_profit_removal,
    worst_removal,
)
from logic.src.policies.helpers.operators.solution_initialization.greedy_si import build_greedy_routes
from logic.src.policies.route_construction.meta_heuristics.ant_colony_optimization_k_sparse.params import (
    KSACOParams,
)
from logic.src.policies.route_construction.meta_heuristics.artificial_bee_colony.params import ABCParams


class ABCSolver:
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
        self.rng = random.Random(params.seed) if params.seed is not None else random.Random()

        # Initialize Local Search once to cache neighbor list
        aco_params = KSACOParams(
            local_search_iterations=self.params.local_search_iterations,
            vrpp=self.params.vrpp,
            profit_aware_operators=self.params.profit_aware_operators,
            seed=params.seed,
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
        Run ABC and return the best solution found.

        Returns:
            Tuple of (routes, profit, cost).
        """
        if self.n_nodes == 0:
            return [], 0.0, 0.0

        start = time.process_time()

        # Initialise food sources (employed bees)
        sources = [self._new_source() for _ in range(self.params.n_sources)]
        profits = [self._evaluate(s) for s in sources]
        trials = [0] * self.params.n_sources

        best_idx = int(np.argmax(profits))
        best_routes = copy.deepcopy(sources[best_idx])
        best_profit = profits[best_idx]
        best_cost = self._cost(best_routes)

        for iteration in range(self.params.max_iterations):
            if self.params.time_limit > 0 and time.process_time() - start > self.params.time_limit:
                break

            # --- Employed bee phase ---
            for i in range(self.params.n_sources):
                # Select a random peer to guide the interpolation
                peer_idx = self.rng.choice([x for x in range(self.params.n_sources) if x != i])
                neighbour = self._perturb(sources[i], sources[peer_idx])
                nb_profit = self._evaluate(neighbour)

                if nb_profit > profits[i]:
                    sources[i] = neighbour
                    profits[i] = nb_profit
                    trials[i] = 0
                else:
                    trials[i] += 1

            # --- Onlooker bee phase ---
            # Fitness for roulette wheel (Karaboga 2005: P_i = fit_i / Sum fit_j)
            # We ensure fitness is non-negative and represents quality
            min_p = min(profits)
            shifted = [max(0.0, p - min_p) + 1e-9 for p in profits]
            total_fit = sum(shifted)
            probs = [s / total_fit for s in shifted]

            for _ in range(self.params.n_sources):
                # Select source i based on probability P_i
                i = self._roulette(probs, self.rng)

                # Perturb and evaluate
                peer_idx = self.rng.choice([x for x in range(self.params.n_sources) if x != i])
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
                    sources[i] = self._build_random_solution()
                    profits[i] = self._evaluate(sources[i])
                    trials[i] = 0

            getattr(self, "_viz_record", lambda **k: None)(
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
        """
        Creates a new food source (initial solution).
        """
        return self._build_random_solution()

    def _build_random_solution(self) -> List[List[int]]:
        """
        Builds a random initial solution using greedy constructive heuristic.
        """
        return build_greedy_routes(
            dist_matrix=self.dist_matrix,
            wastes=self.wastes,
            capacity=self.capacity,
            R=self.R,
            C=self.C,
            mandatory_nodes=self.mandatory_nodes,
            rng=self.rng,
        )

    def _perturb(self, current: List[List[int]], peer: List[List[int]]) -> List[List[int]]:
        """
        Cross-solution interpolation: extracts nodes from a peer and injects
        them into the current solution, mimicking the v_ij = x_ij + φ(x_ij - x_kj) equation.
        """
        if not current or not peer:
            return copy.deepcopy(current)

        n = max(3, self.params.n_removal)
        use_profit = self.params.profit_aware_operators
        expand_pool = self.params.vrpp

        peer_nodes = [node for route in peer for node in route]
        if not peer_nodes:
            return copy.deepcopy(current)

        # Take a random subset of nodes from peer
        selected_peer_nodes = self.rng.sample(peer_nodes, min(n, len(peer_nodes)))

        current_copy = copy.deepcopy(current)
        # Remove these nodes from current
        for route in current_copy:
            for node in selected_peer_nodes:
                if node in route:
                    route.remove(node)

        current_copy = [r for r in current_copy if r]

        # Select operators based on profit_aware_operators flag
        try:
            if use_profit:
                removal_op = self.rng.choice([random_removal, worst_profit_removal])
                reinsert_op = self.rng.choice([greedy_profit_insertion, regret_2_profit_insertion])
                if removal_op == random_removal:
                    partial, removed = random_removal(current_copy, n, rng=self.rng)
                else:
                    partial, removed = worst_profit_removal(
                        current_copy, n, self.dist_matrix, self.wastes, R=self.R, C=self.C
                    )

                to_insert = sorted(list(set(selected_peer_nodes + removed)))
                if reinsert_op == greedy_profit_insertion:
                    repaired = greedy_profit_insertion(
                        partial,
                        to_insert,
                        self.dist_matrix,
                        self.wastes,
                        self.capacity,
                        self.R,
                        self.C,
                        mandatory_nodes=self.mandatory_nodes,
                        expand_pool=expand_pool,
                    )
                else:
                    repaired = regret_2_profit_insertion(
                        partial,
                        to_insert,
                        self.dist_matrix,
                        self.wastes,
                        self.capacity,
                        self.R,
                        self.C,
                        mandatory_nodes=self.mandatory_nodes,
                        expand_pool=expand_pool,
                        noise=0.0,
                    )
            else:
                removal_op_name = self.rng.choice(["random", "worst"])
                reinsert_op_name = self.rng.choice(["greedy", "regret"])
                if removal_op_name == "random":
                    partial, removed = random_removal(current_copy, n, rng=self.rng)
                else:
                    partial, removed = worst_removal(current_copy, n, self.dist_matrix)

                to_insert = sorted(list(set(selected_peer_nodes + removed)))
                if reinsert_op_name == "greedy":
                    repaired = greedy_insertion(
                        partial,
                        to_insert,
                        self.dist_matrix,
                        self.wastes,
                        self.capacity,
                        mandatory_nodes=self.mandatory_nodes,
                        expand_pool=expand_pool,
                    )
                else:
                    repaired = regret_2_insertion(
                        partial,
                        to_insert,
                        self.dist_matrix,
                        self.wastes,
                        self.capacity,
                        mandatory_nodes=self.mandatory_nodes,
                        expand_pool=expand_pool,
                        noise=0.0,
                    )

            # Apply comprehensive local search (reusing instance)
            return self.ls.optimize(repaired)
        except Exception:
            return copy.deepcopy(current)

    @staticmethod
    def _roulette(probs: List[float], rng: random.Random) -> int:
        """
        Roulette-wheel selection.
        """
        r = rng.random()
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
