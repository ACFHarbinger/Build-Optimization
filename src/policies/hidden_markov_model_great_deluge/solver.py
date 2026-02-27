"""
Hidden Markov Model + Great Deluge (HMM-GD) hyper-heuristic for VRPP.

This online-learning approach treats the sequence of applied Low-Level
Heuristics (LLHs) as a Markov chain.  The system observes the objective
change after each LLH application and classifies the current search state
as one of three hidden states: improving (state 0), stagnating (state 1),
or escaping from a local optimum (state 2).

Transition probabilities govern which LLH is most likely to be beneficial
given the current state.  These probabilities are updated online via a
simplified Baum-Welch-like rule based on the observed profit change.

The Great Deluge acceptance criterion accepts candidate solutions whose
profit exceeds a linearly falling water level, providing a deterministic
escape from local optima without requiring temperature tuning.

Reference:
    Survey §"Hyper-Heuristics" — HMM + Great Deluge for selection hyper-heuristics.
"""

import copy
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.tracking.viz_mixin import PolicyVizMixin

from ..operators.destroy_operators import cluster_removal, random_removal, worst_removal
from ..operators.repair_operators import greedy_insertion, regret_2_insertion
from .params import HMMGDParams

# HMM states
_STATE_IMPROVING = 0
_STATE_STAGNATING = 1
_STATE_ESCAPING = 2
_N_STATES = 3


class HMMGDSolver(PolicyVizMixin):
    """
    HMM + Great Deluge hyper-heuristic solver for VRPP.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: HMMGDParams,
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
        self.n_llh = params.n_llh

        # LLH pool
        self._llh_pool = [
            self._llh0,
            self._llh1,
            self._llh2,
            self._llh3,
            self._llh4,
        ]

        # HMM transition matrix A[state] -> probability over LLHs
        # Initialised uniformly
        self._A: np.ndarray = np.ones((_N_STATES, self.n_llh)) / self.n_llh

        # LLH performance accumulators per state
        self._llh_hits: np.ndarray = np.zeros((_N_STATES, self.n_llh))
        self._llh_total: np.ndarray = np.ones((_N_STATES, self.n_llh))  # avoid /0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def solve(self) -> Tuple[List[List[int]], float, float]:
        """
        Run HMM-GD and return the best solution found.

        Returns:
            Tuple of (routes, profit, cost).
        """
        if self.n_nodes == 0:
            return [], 0.0, 0.0

        start = time.time()

        # Initial solution
        routes = self._random_solution()
        profit = self._evaluate(routes)
        best_routes = copy.deepcopy(routes)
        best_profit = profit
        best_cost = self._cost(best_routes)

        # Great Deluge for maximization: water level starts *below* initial profit
        # and slowly rises. We accept any move that is better than the water level.
        water_level = (
            best_profit * (1.0 - self.params.flood_margin) if best_profit > 0 else -abs(self.params.flood_margin)
        )

        # Current HMM state
        state = _STATE_IMPROVING
        stagnation_count = 0

        for iteration in range(self.params.max_iterations):
            if time.time() - start > self.params.time_limit:
                break

            # Select LLH from HMM transition probabilities for current state
            llh_probs = self._A[state]
            llh_idx = self._sample_llh(llh_probs)
            llh = self._llh_pool[llh_idx]

            # Apply LLH
            try:
                new_routes = llh(routes, self.params.n_removal)

                # Apply 2-opt after each LLH application
                from logic.src.policies.local_search.local_search_aco import ACOLocalSearch

                ls = ACOLocalSearch(self.dist_matrix, self.wastes, self.capacity, self.R, self.C, self.params)
                new_routes = ls.optimize(new_routes)

                new_profit = self._evaluate(new_routes)
            except Exception:
                new_routes = routes
                new_profit = profit

            delta = new_profit - profit

            # --- Great Deluge acceptance (Maximization) ---
            # Accept if profit is better than the rising water level
            accepted = new_profit >= water_level

            if accepted:
                routes = new_routes
                profit = new_profit

                if profit > best_profit:
                    best_routes = copy.deepcopy(routes)
                    best_profit = profit
                    best_cost = self._cost(best_routes)

            # --- HMM state transition ---
            prev_state = state
            if delta > 1e-9:
                state = _STATE_IMPROVING
                stagnation_count = 0
            elif stagnation_count > 10:
                state = _STATE_ESCAPING
                stagnation_count = 0
            else:
                state = _STATE_STAGNATING
                stagnation_count += 1

            # --- Online HMM update ---
            # Record LLH performance: "hit" if improvement, "miss" otherwise
            self._llh_total[prev_state][llh_idx] += 1
            if delta > 0:
                self._llh_hits[prev_state][llh_idx] += 1

            # Update transition probabilities with online learning
            success_rate = self._llh_hits[prev_state][llh_idx] / self._llh_total[prev_state][llh_idx]
            lr = self.params.learning_rate
            self._A[prev_state][llh_idx] = (1.0 - lr) * self._A[prev_state][llh_idx] + lr * success_rate
            # Re-normalise row
            row_sum = self._A[prev_state].sum()
            if row_sum > 1e-9:
                self._A[prev_state] /= row_sum
            else:
                self._A[prev_state] = np.ones(self.n_llh) / self.n_llh

            # Increase water level (flood rises)
            water_level += self.params.rain_speed * abs(best_profit + 1e-9)

            self._viz_record(
                iteration=iteration,
                best_profit=best_profit,
                best_cost=best_cost,
                water_level=water_level,
                hmm_state=state,
                llh_selected=llh_idx,
            )

        return best_routes, best_profit, best_cost

    # ------------------------------------------------------------------
    # LLH pool
    # ------------------------------------------------------------------

    def _llh0(self, routes: List[List[int]], n: int) -> List[List[int]]:
        """L0: random_removal + greedy_insertion."""
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
        """L1: worst_removal + regret_2_insertion."""
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
        """L2: cluster_removal + greedy_insertion."""
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
        """L3: worst_removal + greedy_insertion."""
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
        """L4: random_removal + regret_2_insertion."""
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

    @staticmethod
    def _sample_llh(probs: np.ndarray) -> int:
        """Sample an LLH index from the probability distribution."""
        r = random.random()
        cumulative = 0.0
        for i, p in enumerate(probs):
            cumulative += p
            if r <= cumulative:
                return i
        return len(probs) - 1

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
