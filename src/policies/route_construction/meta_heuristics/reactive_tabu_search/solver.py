"""
Reactive Tabu Search (RTS) for VRPP.

Tabu Search uses short-term memory to forbid recently visited moves,
preventing cyclical revisitation.  Reactive Tabu Search enhances this
with hash-based cycle detection: when configuration hashes repeat,
the tabu tenure is increased to amplify diversification; during long
non-cycling periods, tenure contracts for intensive exploitation.

Reference:
    Battiti, R., & Tecchiolli, G. "The Reactive Tabu Search", 1994.
"""

import copy
import random
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple, cast

import numpy as np

from logic.src.policies.helpers.operators import (
    cluster_removal,
    greedy_insertion,
    greedy_profit_insertion,
    random_removal,
    regret_2_insertion,
    regret_2_profit_insertion,
    shaw_profit_removal,
    shaw_removal,
    string_removal,
    worst_profit_removal,
    worst_removal,
)
from logic.src.policies.helpers.operators.solution_initialization.nearest_neighbor_si import build_nn_routes

from .params import RTSParams


class RTSSolver:
    """
    Reactive Tabu Search solver for VRPP.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: RTSParams,
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
        self.random = random.Random(params.seed) if params.seed is not None else random.Random()

        self._llh_pool = [
            self._llh0,
            self._llh1,
            self._llh2,
            self._llh3,
            self._llh4,
            self._llh5,
            self._llh6,
        ]

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def solve(self) -> Tuple[List[List[int]], float, float]:
        """
        Run Reactive Tabu Search.

        Returns:
            Tuple of (routes, profit, cost).
        """
        if self.n_nodes == 0:
            return [], 0.0, 0.0

        start = time.process_time()

        routes = self._build_initial_solution()
        profit = self._evaluate(routes)
        best_routes = copy.deepcopy(routes)
        best_profit = profit

        # Setup modular acceptance criterion (Aspiration folding)
        assert self.params.acceptance_criterion is not None
        self.params.acceptance_criterion.setup(profit)

        tenure = self.params.initial_tenure
        # Tabu list: deque of (llh_idx, solution_hash) pairs
        tabu_list: Deque[Tuple[int, int]] = deque(maxlen=self.params.max_tenure)
        # Hash history for cycle detection
        hash_history: Dict[int, int] = {}
        no_repeat_count = 0

        for iteration in range(self.params.max_iterations):
            if self.params.time_limit > 0 and time.process_time() - start > self.params.time_limit:
                break

            # Try all LLHs and pick best non-tabu (or aspiration override)
            best_candidate = None
            best_cand_profit = float("-inf")
            best_cand_llh = -1

            for llh_idx in range(self.params.n_llh):
                try:
                    cand = self._llh_pool[llh_idx](copy.deepcopy(routes), self.params.n_removal)
                    cand_profit = self._evaluate(cand)
                except Exception:
                    continue

                sol_hash = self._hash_routes(cand)
                is_tabu = any(h == sol_hash for _, h in tabu_list)

                # Modular Aspiration: Consult criterion if move is tabu
                if is_tabu and not cast(Any, self.params.acceptance_criterion).accept(
                    current_obj=profit, candidate_obj=cand_profit, is_tabu=True, it=iteration
                ):
                    continue

                if cand_profit > best_cand_profit:
                    best_candidate = cand
                    best_cand_profit = cand_profit
                    best_cand_llh = llh_idx

            if best_candidate is None:
                # All moves tabu — force a random one
                llh_idx = self.random.randint(0, self.params.n_llh - 1)
                try:
                    best_candidate = self._llh_pool[llh_idx](copy.deepcopy(routes), self.params.n_removal)
                    best_cand_profit = self._evaluate(best_candidate)
                    best_cand_llh = llh_idx
                except Exception:
                    continue

            # Move to best candidate
            routes = best_candidate
            profit = best_cand_profit
            sol_hash = self._hash_routes(routes)

            # Add to tabu list
            tabu_list.append((best_cand_llh, sol_hash))
            # Trim to current tenure
            while len(tabu_list) > tenure:
                tabu_list.popleft()

            if profit > best_profit:
                best_routes = copy.deepcopy(routes)
                best_profit = profit

            # Step the criterion after move acceptance
            cast(Any, self.params.acceptance_criterion).step(
                current_obj=profit,
                candidate_obj=best_cand_profit,
                accepted=True,
                it=iteration,
            )

            # Reactive tenure adjustment via cycle detection (Battiti & Tecchiolli 1994)
            if sol_hash in hash_history:
                # Cycle detected: increase tenure
                # The paper suggests multiplying by a constant (e.g., 1.1)

                tenure = min(
                    self.params.max_tenure, max(self.params.min_tenure, int(tenure * self.params.tenure_increase))
                )

                # Reset the 'moving window' check
                no_repeat_count = 0

                # Escape mechanism: if we stay in a same cycle length/state too long
                # (Optional: specialized diversification can be triggered here)
            else:
                no_repeat_count += 1
                # If we go 2*tenure iterations without any repetition, we contract
                if no_repeat_count > 2 * tenure:
                    tenure = max(self.params.min_tenure, int(tenure * self.params.tenure_decrease))
                    no_repeat_count = 0

            # Update hash history with the latest iteration seen
            hash_history[sol_hash] = iteration

            getattr(self, "_viz_record", lambda **k: None)(
                iteration=iteration,
                best_profit=best_profit,
                best_cost=self._cost(best_routes),
                tenure=tenure,
            )

        best_cost = self._cost(best_routes)
        return best_routes, best_profit, best_cost

    # ------------------------------------------------------------------
    # LLH pool
    # ------------------------------------------------------------------

    def _llh0(self, routes: List[List[int]], n: int) -> List[List[int]]:
        if self.params.profit_aware_operators:
            partial, removed = random_removal(routes, n, self.random)
            return greedy_profit_insertion(
                partial,
                removed,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                self.R,
                self.C,
                self.mandatory_nodes,
                self.params.vrpp,
            )
        else:
            partial, removed = random_removal(routes, n, self.random)
            return greedy_insertion(
                partial,
                removed,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                mandatory_nodes=self.mandatory_nodes,
                expand_pool=self.params.vrpp,
            )

    def _llh1(self, routes: List[List[int]], n: int) -> List[List[int]]:
        if self.params.profit_aware_operators:
            partial, removed = worst_profit_removal(routes, n, self.dist_matrix, self.wastes, self.R, self.C)
            return regret_2_profit_insertion(
                partial,
                removed,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                self.R,
                self.C,
                self.mandatory_nodes,
                self.params.vrpp,
            )
        else:
            partial, removed = worst_removal(routes, n, self.dist_matrix)
            return regret_2_insertion(
                partial,
                removed,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                mandatory_nodes=self.mandatory_nodes,
                expand_pool=self.params.vrpp,
            )

    def _llh2(self, routes: List[List[int]], n: int) -> List[List[int]]:
        if self.params.profit_aware_operators:
            partial, removed = cluster_removal(routes, n, self.dist_matrix, self.nodes)
            return greedy_profit_insertion(
                partial,
                removed,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                self.R,
                self.C,
                self.mandatory_nodes,
                self.params.vrpp,
            )
        else:
            partial, removed = cluster_removal(routes, n, self.dist_matrix, self.nodes)
            return greedy_insertion(
                partial,
                removed,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                mandatory_nodes=self.mandatory_nodes,
                expand_pool=self.params.vrpp,
            )

    def _llh3(self, routes: List[List[int]], n: int) -> List[List[int]]:
        if self.params.profit_aware_operators:
            partial, removed = worst_profit_removal(routes, n, self.dist_matrix, self.wastes, self.R, self.C)
            return greedy_profit_insertion(
                partial,
                removed,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                self.R,
                self.C,
                self.mandatory_nodes,
                self.params.vrpp,
            )
        else:
            partial, removed = worst_removal(routes, n, self.dist_matrix)
            return greedy_insertion(
                partial,
                removed,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                mandatory_nodes=self.mandatory_nodes,
                expand_pool=self.params.vrpp,
            )

    def _llh4(self, routes: List[List[int]], n: int) -> List[List[int]]:
        partial, removed = random_removal(routes, n, self.random)
        if self.params.profit_aware_operators:
            return regret_2_profit_insertion(
                partial,
                removed,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                self.R,
                self.C,
                self.mandatory_nodes,
                self.params.vrpp,
            )
        else:
            return regret_2_insertion(
                partial,
                removed,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                mandatory_nodes=self.mandatory_nodes,
                expand_pool=self.params.vrpp,
            )

    def _llh5(self, routes: List[List[int]], n: int) -> List[List[int]]:
        if self.params.profit_aware_operators:
            partial, removed = shaw_profit_removal(
                routes, n, self.dist_matrix, self.wastes, self.R, self.C, rng=self.random
            )
            return regret_2_profit_insertion(
                partial,
                removed,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                self.R,
                self.C,
                self.mandatory_nodes,
                self.params.vrpp,
            )
        else:
            partial, removed = shaw_removal(routes, n, self.dist_matrix, wastes=self.wastes)
            return regret_2_insertion(
                partial,
                removed,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                mandatory_nodes=self.mandatory_nodes,
                expand_pool=self.params.vrpp,
            )

    def _llh6(self, routes: List[List[int]], n: int) -> List[List[int]]:
        partial, removed = string_removal(routes, n, self.dist_matrix, rng=self.random)
        if self.params.profit_aware_operators:
            return greedy_profit_insertion(
                partial,
                removed,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                self.R,
                self.C,
                self.mandatory_nodes,
                self.params.vrpp,
            )
        else:
            return greedy_insertion(
                partial,
                removed,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                mandatory_nodes=self.mandatory_nodes,
                expand_pool=self.params.vrpp,
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _hash_routes(self, routes: List[List[int]]) -> int:
        """Compute a hash of the route configuration for cycle detection."""
        return hash(tuple(tuple(r) for r in routes))

    def _build_initial_solution(self) -> List[List[int]]:
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
