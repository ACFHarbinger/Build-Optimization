"""
Sequence-based Selection Hyper-Heuristic (SS-HH) for VRPP.

This solver implements Algorithm 1 from Kheiri (2014): "Heuristic Sequence
Selection for Inventory Routing Problem".

The core idea is to build *sequences* of Low-Level Heuristics (LLHs) before
applying them. Two HMM-style score matrices drive the selection:

- TMatrix[h_prev][h_cur]: transition score from the previous LLH to the next.
  Roulette-wheel selection over TMatrix rows determines which LLH follows.

- ASMatrix[h_cur][AS]: acceptance-strategy score.  AS=0 means "extend the
  current sequence"; AS=1 means "apply the sequence now".

When a sequence improves the global best solution, all relevant TMatrix and
ASMatrix entries are rewarded with a Δ_norm-weighted increment (the normalised
relative improvement), so that sequences producing larger gains receive
proportionally larger credit.

Move acceptance follows Eq. 4 from the paper: a candidate is accepted if its
quality is ≤ the current candidate, or if it is within a time-decaying
threshold of the best solution.

Reference:
    Kheiri, A. "Heuristic Sequence Selection for Inventory Routing Problem", 2014.
    Bibliography: bibliography/Sequence-based_Selection_Hyper-Heuristic.pdf
"""

import contextlib
import copy
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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

from .params import SSHHParams


@dataclass
class _SeqEntry:
    """One entry in the heuristic sequence (Algorithm 1, line 5).

    Attributes:
        h_prev: Index of the previous LLH.
        h_cur: Index of the current LLH.
        AS: Acceptance strategy flag (0 = extend sequence, 1 = apply).
    """

    h_prev: int
    h_cur: int
    AS: int


class SSHHSolver:
    """
    Sequence-based Selection Hyper-Heuristic (SS-HH) solver for VRPP.

    Implements Algorithm 1 from Kheiri (2014).  Key components:

    - **TMatrix[n_llh][n_llh]**: transition scores between LLHs.  Roulette-wheel
      selection via Eq. 2.  Initialised to 1.
    - **ASMatrix[n_llh][2]**: per-LLH acceptance strategy scores (AS=0 extend,
      AS=1 apply).  Roulette-wheel via Eq. 3.  Initialised to 1.
    - **Sequence building**: each step either extends the sequence (AS=0) or
      triggers application of the entire accumulated sequence (AS=1).
    - **Move acceptance**: threshold-based criterion (Eq. 4).
    - **Score updates**: Δ_norm-weighted increments when the sequence improves
      the global best.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: SSHHParams,
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
        self.random = random.Random(params.seed) if params.seed is not None else random.Random()

        # LLH pool
        self._llh_pool = [
            self._llh0,
            self._llh1,
            self._llh2,
            self._llh3,
            self._llh4,
            self._llh5,
            self._llh6,
        ]

        # Algorithm 1, lines 6-11: initialise HMM matrices to 1
        self._TMatrix: np.ndarray = np.ones((self.n_llh, self.n_llh), dtype=np.float64)
        self._ASMatrix: np.ndarray = np.ones((self.n_llh, 2), dtype=np.float64)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def solve(self) -> Tuple[List[List[int]], float, float]:
        """
        Run SS-HH (Algorithm 1) and return the best solution found.

        Returns:
            Tuple of (routes, profit, cost).
        """
        if self.n_nodes == 0:
            return [], 0.0, 0.0

        start = time.process_time()

        # Algorithm 1, line 12: construct initial solution
        routes = self._build_initial_solution()
        profit = self._evaluate(routes)
        candidate_routes = copy.deepcopy(routes)
        candidate_profit = profit
        best_routes = copy.deepcopy(routes)
        best_profit = profit
        best_cost = self._cost(best_routes)

        # Algorithm 1, line 13: select random initial heuristic
        h_current = self.random.randrange(self.n_llh)

        # Sequence accumulator (Algorithm 1, line 5)
        seq: List[_SeqEntry] = []

        for iteration in range(self.params.max_iterations):
            if self.params.time_limit > 0 and time.process_time() - start > self.params.time_limit:
                break

            # Algorithm 1, line 15
            h_previous = h_current

            # Algorithm 1, line 16: roulette-wheel over TMatrix[h_prev]
            h_current = self._roulette_wheel_row(self._TMatrix[h_previous])

            # Algorithm 1, line 17: roulette-wheel over ASMatrix[h_current]
            AS = self._roulette_wheel_row(self._ASMatrix[h_current])

            # Algorithm 1, line 18: record the step
            seq.append(_SeqEntry(h_prev=h_previous, h_cur=h_current, AS=AS))

            # Algorithm 1, line 19: if AS == 1, apply the full sequence
            if AS == 1:
                # Algorithm 1, lines 20-22: apply each heuristic in SEQ
                s_new_routes = copy.deepcopy(candidate_routes)
                for entry in seq:
                    with contextlib.suppress(Exception):
                        s_new_routes = self._llh_pool[entry.h_cur](s_new_routes, self.params.n_removal)

                s_new_profit = self._evaluate(s_new_routes)

                # Algorithm 1, line 23: move acceptance (Eq. 4)
                elapsed = time.process_time() - start
                accepted = self._accept(s_new_profit, candidate_profit, best_profit, elapsed)

                if accepted:
                    # Algorithm 1, line 24
                    candidate_routes = s_new_routes
                    candidate_profit = s_new_profit

                    # Algorithm 1, lines 25-29: update best & reward scores
                    if s_new_profit > best_profit:
                        # Δ_norm-weighted reward (image fix #2)
                        delta_norm = (
                            (s_new_profit - best_profit) / abs(best_profit) if abs(best_profit) > 1e-12 else 1.0
                        )
                        reward = max(delta_norm, 1.0)  # at least +1

                        best_routes = copy.deepcopy(s_new_routes)
                        best_profit = s_new_profit
                        best_cost = self._cost(best_routes)

                        # Algorithm 1, lines 27-29: reward all entries in SEQ
                        for entry in seq:
                            self._TMatrix[entry.h_prev][entry.h_cur] += reward
                            self._ASMatrix[entry.h_cur][entry.AS] += reward

                # Algorithm 1, line 30: clear SEQ
                seq.clear()

            getattr(self, "_viz_record", lambda **k: None)(
                iteration=iteration,
                best_profit=best_profit,
                best_cost=best_cost,
                seq_length=len(seq),
                llh_selected=h_current,
            )

        # Algorithm 1, line 31
        return best_routes, best_profit, best_cost

    # ------------------------------------------------------------------
    # Move acceptance (Eq. 4)
    # ------------------------------------------------------------------

    def _accept(
        self,
        new_profit: float,
        candidate_profit: float,
        best_profit: float,
        elapsed: float,
    ) -> bool:
        """Threshold-based acceptance criterion from Eq. 4.

        Accept if either:
        - new_profit ≥ candidate_profit (quality ≤ in minimisation terms), or
        - new_profit > best_profit - T * |best_profit|  (within threshold)

        The threshold T decays linearly with elapsed time:
            T = threshold_infeasible                     if best not feasible
            T = threshold_feasible_base + threshold_decay * (1 - elapsed/limit)

        In this VRPP context all solutions are feasible (profit-collecting),
        so we always use the feasible formula.
        """
        # Direct improvement
        if new_profit >= candidate_profit:
            return True

        # Threshold acceptance
        t_limit = self.params.time_limit if self.params.time_limit > 0 else 1.0
        time_ratio = min(elapsed / t_limit, 1.0)
        T = self.params.threshold_feasible_base + self.params.threshold_decay_rate * (1.0 - time_ratio)

        threshold = best_profit - T * abs(best_profit) if abs(best_profit) > 1e-12 else best_profit - T
        return new_profit > threshold

    # ------------------------------------------------------------------
    # Roulette-wheel selection (Eq. 2 / Eq. 3)
    # ------------------------------------------------------------------

    def _roulette_wheel_row(self, scores: np.ndarray) -> int:
        """Roulette-wheel selection over a row of raw scores.

        P(j) = scores[j] / Σ_k scores[k]   (Eq. 2 for TMatrix, Eq. 3 for ASMatrix)
        """
        total = scores.sum()
        if total <= 0:
            return self.random.randrange(len(scores))
        r = self.random.random() * total
        cumulative = 0.0
        for i, s in enumerate(scores):
            cumulative += s
            if r <= cumulative:
                return i
        return len(scores) - 1

    # ------------------------------------------------------------------
    # LLH pool
    # ------------------------------------------------------------------

    def _llh0(self, routes: List[List[int]], n: int) -> List[List[int]]:
        """L0: random_removal + greedy_insertion."""
        use_profit = self.params.profit_aware_operators
        expand_pool = self.params.vrpp

        p, r = random_removal(routes, n, self.random)
        if use_profit:
            return greedy_profit_insertion(
                p, r, self.dist_matrix, self.wastes, self.capacity, self.R, self.C, expand_pool=expand_pool
            )
        return greedy_insertion(
            p,
            r,
            self.dist_matrix,
            self.wastes,
            self.capacity,
            mandatory_nodes=self.mandatory_nodes,
            expand_pool=expand_pool,
        )

    def _llh1(self, routes: List[List[int]], n: int) -> List[List[int]]:
        """L1: worst_removal + regret_2_insertion."""
        use_profit = self.params.profit_aware_operators
        expand_pool = self.params.vrpp

        if use_profit:
            p, r = worst_profit_removal(routes, n, self.dist_matrix, self.wastes, self.R, self.C)
            return regret_2_profit_insertion(
                p, r, self.dist_matrix, self.wastes, self.capacity, self.R, self.C, expand_pool=expand_pool
            )
        p, r = worst_removal(routes, n, self.dist_matrix)
        return regret_2_insertion(
            p,
            r,
            self.dist_matrix,
            self.wastes,
            self.capacity,
            mandatory_nodes=self.mandatory_nodes,
            expand_pool=expand_pool,
        )

    def _llh2(self, routes: List[List[int]], n: int) -> List[List[int]]:
        """L2: cluster_removal + greedy_insertion."""
        use_profit = self.params.profit_aware_operators
        expand_pool = self.params.vrpp

        p, r = cluster_removal(routes, n, self.dist_matrix, self.nodes)
        if use_profit:
            return greedy_profit_insertion(
                p, r, self.dist_matrix, self.wastes, self.capacity, self.R, self.C, expand_pool=expand_pool
            )
        return greedy_insertion(
            p,
            r,
            self.dist_matrix,
            self.wastes,
            self.capacity,
            mandatory_nodes=self.mandatory_nodes,
            expand_pool=expand_pool,
        )

    def _llh3(self, routes: List[List[int]], n: int) -> List[List[int]]:
        """L3: worst_removal + greedy_insertion."""
        use_profit = self.params.profit_aware_operators
        expand_pool = self.params.vrpp

        if use_profit:
            p, r = worst_profit_removal(routes, n, self.dist_matrix, self.wastes, self.R, self.C)
            return greedy_profit_insertion(
                p, r, self.dist_matrix, self.wastes, self.capacity, self.R, self.C, expand_pool=expand_pool
            )
        p, r = worst_removal(routes, n, self.dist_matrix)
        return greedy_insertion(
            p,
            r,
            self.dist_matrix,
            self.wastes,
            self.capacity,
            mandatory_nodes=self.mandatory_nodes,
            expand_pool=expand_pool,
        )

    def _llh4(self, routes: List[List[int]], n: int) -> List[List[int]]:
        """L4: random_removal + regret_2_insertion."""
        use_profit = self.params.profit_aware_operators
        expand_pool = self.params.vrpp
        p, r = random_removal(routes, n, self.random)
        if use_profit:
            return regret_2_profit_insertion(
                p, r, self.dist_matrix, self.wastes, self.capacity, self.R, self.C, expand_pool=expand_pool
            )
        return regret_2_insertion(
            p,
            r,
            self.dist_matrix,
            self.wastes,
            self.capacity,
            mandatory_nodes=self.mandatory_nodes,
            expand_pool=expand_pool,
        )

    def _llh5(self, routes: List[List[int]], n: int) -> List[List[int]]:
        """L5: shaw_removal + greedy_insertion."""
        use_profit = self.params.profit_aware_operators
        expand_pool = self.params.vrpp

        if use_profit:
            p, r = shaw_profit_removal(routes, n, self.dist_matrix, self.wastes, self.R, self.C)
            return greedy_profit_insertion(
                p, r, self.dist_matrix, self.wastes, self.capacity, self.R, self.C, expand_pool=expand_pool
            )
        else:
            p, r = shaw_removal(routes, n, self.dist_matrix)
            return greedy_insertion(
                p,
                r,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                mandatory_nodes=self.mandatory_nodes,
                expand_pool=expand_pool,
            )

    def _llh6(self, routes: List[List[int]], n: int) -> List[List[int]]:
        """L6: string_removal + regret_2_insertion."""
        use_profit = self.params.profit_aware_operators
        expand_pool = self.params.vrpp

        p, r = string_removal(routes, n, self.dist_matrix, rng=self.random)
        if use_profit:
            return regret_2_profit_insertion(
                p, r, self.dist_matrix, self.wastes, self.capacity, self.R, self.C, expand_pool=expand_pool
            )
        return regret_2_insertion(
            p,
            r,
            self.dist_matrix,
            self.wastes,
            self.capacity,
            mandatory_nodes=self.mandatory_nodes,
            expand_pool=expand_pool,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_initial_solution(self) -> List[List[int]]:
        """Algorithm 1, line 12: construct initial solution."""
        return build_nn_routes(
            nodes=self.nodes,
            mandatory_nodes=self.mandatory_nodes,
            wastes=self.wastes,
            capacity=self.capacity,
            dist_matrix=self.dist_matrix,
            R=self.R,
            C=self.C,
        )

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
