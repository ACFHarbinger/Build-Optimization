"""
Hidden Markov Model + Great Deluge (HMM-GD) hyper-heuristic for Build Optimization.

Online-learning selection hyper-heuristic for choosing LLHs.
Acceptance via Great Deluge (rising water level).
"""

import random
import time
from typing import Tuple

import numpy as np

from core.problem import BuildProblem
from tracking.viz_mixin import PolicyVizMixin

from ..operators.destroy_operators import cluster_removal, random_removal, worst_removal
from ..operators.repair_operators import greedy_blink_insertion, greedy_insertion, regret_2_insertion
from .params import HMMGDParams

# HMM states
_STATE_IMPROVING = 0
_STATE_STAGNATING = 1
_STATE_ESCAPING = 2
_N_STATES = 3


class HMMGDSolver(PolicyVizMixin):
    """
    HMM + Great Deluge hyper-heuristic solver for Build Optimization.
    """

    def __init__(
        self,
        problem: BuildProblem,
        budget: float,
        params: HMMGDParams,
    ):
        self.problem = problem
        self.budget = budget
        self.params = params
        self.n_llh = params.n_llh

        # LLH pool
        self._llh_pool = [
            self._llh_greedy,
            self._llh_regret,
            self._llh_blink,
        ]
        # Adjust n_llh to actual pool size if different
        self.n_llh = len(self._llh_pool)

        # HMM transition matrix A[state] -> probability over LLHs
        self._A: np.ndarray = np.ones((_N_STATES, self.n_llh)) / self.n_llh

        # LLH performance accumulators per state
        self._llh_hits: np.ndarray = np.zeros((_N_STATES, self.n_llh))
        self._llh_total: np.ndarray = np.ones((_N_STATES, self.n_llh))

    def solve(self) -> Tuple[np.ndarray, float]:
        """
        Run HMM-GD and return the best build found.

        Returns:
            Tuple of (best_build, best_score).
        """
        start = time.time()

        # Initial solution
        current_build = self.problem.greedy_solution()
        current_score = self.problem.evaluate(current_build)

        best_build = current_build.copy()
        best_score = current_score

        # Great Deluge for maximization: water level starts below initial score
        water_level = current_score * (1.0 - self.params.flood_margin)

        # Current HMM state
        state = _STATE_IMPROVING
        stagnation_count = 0

        for iteration in range(self.params.max_iterations):
            if time.time() - start > self.params.time_limit:
                break

            # Select LLH from HMM transition probabilities
            llh_probs = self._A[state]
            llh_idx = self._sample_llh(llh_probs)
            llh = self._llh_pool[llh_idx]

            # Apply LLH
            try:
                new_build = llh(current_build)
                new_score = self.problem.evaluate(new_build)
            except Exception:
                new_build = current_build
                new_score = current_score

            delta = new_score - current_score

            # --- Great Deluge acceptance (Maximization) ---
            accepted = new_score >= water_level

            if accepted:
                current_build = new_build
                current_score = new_score

                if current_score > best_score:
                    best_build = current_build.copy()
                    best_score = current_score

            # --- HMM state transition ---
            prev_state = state
            if delta > 1e-6:
                state = _STATE_IMPROVING
                stagnation_count = 0
            elif stagnation_count > 10:
                state = _STATE_ESCAPING
                stagnation_count = 0
            else:
                state = _STATE_STAGNATING
                stagnation_count += 1

            # --- Online HMM update ---
            self._llh_total[prev_state][llh_idx] += 1
            if delta > 0:
                self._llh_hits[prev_state][llh_idx] += 1

            # Update transition probabilities
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
            water_level += self.params.rain_speed * abs(best_score + 1e-9)

            self._viz_record(
                iteration=iteration,
                best_profit=best_score,
                best_cost=-best_score,
                water_level=water_level,
                hmm_state=state,
                llh_selected=llh_idx,
            )

        return best_build, best_score

    def _sample_llh(self, probs: np.ndarray) -> int:
        r = random.random()
        cumulative = 0.0
        for i, p in enumerate(probs):
            cumulative += p
            if r <= cumulative:
                return i
        return len(probs) - 1

    def _llh_greedy(self, build: np.ndarray) -> np.ndarray:
        partial = random_removal(build, self.params.n_removal, self.problem)
        return greedy_insertion(partial, self.budget, self.problem)

    def _llh_regret(self, build: np.ndarray) -> np.ndarray:
        partial = worst_removal(build, self.params.n_removal, self.problem)
        return regret_2_insertion(partial, self.budget, self.problem)

    def _llh_blink(self, build: np.ndarray) -> np.ndarray:
        partial = cluster_removal(build, self.params.n_removal, self.problem)
        return greedy_blink_insertion(partial, self.budget, self.problem)
