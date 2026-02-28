"""
Hyper-Heuristic Ant Colony Optimization (Hyper-ACO) for Build Optimization.
"""

import time
from typing import List, Optional, Tuple

import numpy as np

from core.problem import BuildProblem
from tracking.viz_mixin import PolicyVizMixin

from .hyper_operators import HYPER_OPERATORS, HyperOperatorContext
from .params import HyperACOParams


class HyperHeuristicACO(PolicyVizMixin):
    """
    Hyper-Heuristic ACO solver for Build Optimization.
    """

    def __init__(
        self,
        problem: BuildProblem,
        budget: float,
        params: Optional[HyperACOParams] = None,
    ):
        self.problem = problem
        self.budget = budget
        self.params = params or HyperACOParams()

        self.operator_names = list(HYPER_OPERATORS.keys())
        self.n_operators = len(self.operator_names)

        # Pheromone matrix: n_operators+1 x n_operators
        self.tau = np.full((self.n_operators + 1, self.n_operators), self.params.tau_0)

        self.eta = np.ones(self.n_operators)
        self.success_counts = np.zeros(self.n_operators)
        self.use_counts = np.ones(self.n_operators)

    def solve(self, initial_build: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Solve Build Optimization using Hyper-ACO.
        """
        best_build = initial_build.copy()
        best_score = self.problem.evaluate(best_build)

        start_time = time.time()
        for iteration in range(self.params.max_iterations):
            if time.time() - start_time > self.params.time_limit:
                break

            ant_solutions = []
            for _ant in range(self.params.n_ants):
                build, score = self.build_solution(initial_build)
                ant_solutions.append((build, score))

            # Pick best of iteration
            ant_solutions.sort(key=lambda x: x[1], reverse=True)
            iter_best_build, iter_best_score = ant_solutions[0]

            if iter_best_score > best_score:
                best_score = iter_best_score
                best_build = iter_best_build.copy()

            self._evaporate_pheromones()
            self._update_heuristics()

            self._viz_record(
                iteration=iteration,
                best_profit=best_score,
                iter_best_profit=iter_best_score,
                tau_mean=float(self.tau.mean()),
                eta_mean=float(self.eta.mean()),
            )

        return best_build, best_score

    def build_solution(self, base_build: np.ndarray) -> Tuple[np.ndarray, float]:
        sequence = self._select_sequence()
        ctx = HyperOperatorContext(
            problem=self.problem,
            budget=self.budget,
            build=base_build.copy(),
        )

        for op_name in sequence:
            op_func = HYPER_OPERATORS.get(op_name)
            if op_func:
                op_idx = self.operator_names.index(op_name)
                self.use_counts[op_idx] += 1
                if op_func(ctx):
                    self.success_counts[op_idx] += 1

        return ctx.build, ctx.score

    def _select_sequence(self) -> List[str]:
        sequence = []
        current_op_idx = self.n_operators  # Start state
        for _ in range(self.params.sequence_length):
            probs = (self.tau[current_op_idx] ** self.params.alpha) * (self.eta**self.params.beta)
            probs /= np.sum(probs)
            next_op_idx = np.random.choice(self.n_operators, p=probs)
            sequence.append(self.operator_names[next_op_idx])
            current_op_idx = next_op_idx
        return sequence

    def _evaporate_pheromones(self):
        self.tau *= 1 - self.params.rho
        np.clip(self.tau, self.params.tau_min, self.params.tau_max, out=self.tau)

    def _update_heuristics(self):
        self.eta = self.success_counts / self.use_counts
        self.eta = np.clip(self.eta, 0.01, 10.0)
