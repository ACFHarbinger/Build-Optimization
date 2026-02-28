"""
Adaptive Large Neighborhood Search (ALNS) policy module for Build Optimization.

This module provides the main entry points for the ALNS metaheuristic,
operating on discrete item-slot allocations.
"""

import math
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.problem import BuildProblem
from policies.operators import (
    cluster_removal,
    greedy_blink_insertion,
    greedy_insertion,
    random_insertion,
    random_removal,
    regret_2_insertion,
    shaw_removal,
    worst_removal,
)
from tracking.viz_mixin import PolicyVizMixin

from .params import ALNSParams


class ALNSSolver(PolicyVizMixin):
    """
    Custom implementation of Adaptive Large Neighborhood Search for Build Optimization.
    """

    def __init__(
        self,
        problem: BuildProblem,
        budget: float,
        params: ALNSParams,
    ):
        """
        Initialize the ALNS solver.
        """
        self.problem = problem
        self.budget = budget
        self.params = params
        self.num_items = problem.num_items
        # self.num_slots is not used locally in __init__ but available in problem

        # Operator registry
        self.destroy_ops = [
            lambda b, n: random_removal(b, n, self.problem),
            lambda b, n: worst_removal(b, n, self.problem),
            lambda b, n: shaw_removal(b, n, self.problem),
            lambda b, n: cluster_removal(b, n, self.problem),
        ]
        self.repair_ops = [
            lambda b: greedy_insertion(b, self.budget, self.problem),
            lambda b: regret_2_insertion(b, self.budget, self.problem),
            lambda b: greedy_blink_insertion(b, self.budget, self.problem),
            lambda b: random_insertion(b, self.budget, self.problem),
        ]

        self.destroy_weights = [1.0] * len(self.destroy_ops)
        self.repair_weights = [1.0] * len(self.repair_ops)

    def _initialize_solve(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """Initialize the solution and metrics for the solve process."""
        initial_build = self.build_initial_solution()
        best_build = initial_build.copy()

        # Calculate initial metrics
        best_score = self.problem.evaluate(best_build)

        return initial_build, best_build, best_score

    def _select_and_apply_operators(self, current_build: np.ndarray) -> Tuple[np.ndarray, int, int]:
        """Select destroy/repair operators and generate a new solution."""
        d_idx = self.select_operator(self.destroy_weights)
        r_idx = self.select_operator(self.repair_weights)

        destroy_op = self.destroy_ops[d_idx]
        repair_op = self.repair_ops[r_idx]

        n_remove = random.randint(
            self.params.min_removal,
            max(
                self.params.min_removal,
                int(self.problem.num_slots * self.params.max_removal_pct),
            ),
        )

        partial_build = destroy_op(current_build.copy(), n_remove)
        new_build = repair_op(partial_build)

        return new_build, d_idx, r_idx

    def _accept_solution(self, current_score: float, new_score: float, T: float) -> bool:
        """Determine whether to accept the new solution based on SA criteria."""
        delta = new_score - current_score
        if delta > 1e-6:
            return True
        else:
            # delta is <= 0
            prob = math.exp(delta / T) if T > 0 else 0
            return random.random() < prob

    def _update_weights(self, d_idx: int, r_idx: int, score: float):
        """Update the weights of the used operators."""
        lambda_decay = 0.8
        self.destroy_weights[d_idx] = lambda_decay * self.destroy_weights[d_idx] + (1 - lambda_decay) * max(0.1, score)
        self.repair_weights[r_idx] = lambda_decay * self.repair_weights[r_idx] + (1 - lambda_decay) * max(0.1, score)

    def solve(self, initial_solution: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
        """
        Run the ALNS algorithm.
        """
        if initial_solution is not None:
            current_build = initial_solution.copy()
            best_build = current_build.copy()
            best_score = self.problem.evaluate(best_build)
        else:
            (
                current_build,
                best_build,
                best_score,
            ) = self._initialize_solve()

        current_score = best_score

        T = self.params.start_temp
        start_time = time.time()

        for _it in range(self.params.max_iterations):
            if time.time() - start_time > self.params.time_limit:
                break

            new_build, d_idx, r_idx = self._select_and_apply_operators(current_build)

            # Check feasibility
            if not self.problem.is_feasible(new_build):
                # Reject infeasible immediately
                score = 0
                accept = False
            else:
                new_score = self.problem.evaluate(new_build)
                accept = self._accept_solution(current_score, new_score, T)
                score = 0

                if accept:
                    current_build = new_build
                    current_score = new_score
                    if new_score > best_score + 1e-6:
                        best_build = new_build.copy()
                        best_score = new_score
                        score = 3
                    else:
                        score = 1

            self._update_weights(d_idx, r_idx, score)
            T *= self.params.cooling_rate

            self._viz_record(
                iteration=_it,
                d_idx=d_idx,
                r_idx=r_idx,
                best_profit=float(best_score),
                current_profit=float(current_score),
                temperature=float(T),
                accepted=int(accept),
                score=score,
            )

        return best_build, best_score

    def select_operator(self, weights: List[float]) -> int:
        """
        Select an operator index based on their weights using roulette wheel selection.
        """
        total = sum(weights)
        r = random.uniform(0, total)
        curr = 0.0
        for i, w in enumerate(weights):
            curr += w
            if curr >= r:
                return i
        return len(weights) - 1

    def build_initial_solution(self) -> np.ndarray:
        """
        Build a basic feasible solution using greedy construction.
        """
        build = np.full(self.problem.num_slots, -1, dtype=int)
        return greedy_insertion(build, self.budget, self.problem)


def run_alns(problem: BuildProblem, budget: float, values: Dict[str, Any], *args: Any) -> Tuple[np.ndarray, float]:
    """
    Main ALNS entry point with dispatching to different algorithm variants.
    """
    params = ALNSParams(
        time_limit=values.get("time_limit", 10),
        max_iterations=values.get("max_iterations", 2000),
        start_temp=values.get("start_temp", 100.0),
        cooling_rate=values.get("cooling_rate", 0.995),
        reaction_factor=values.get("reaction_factor", 0.1),
        min_removal=values.get("min_removal", 1),
        max_removal_pct=values.get("max_removal_pct", 0.3),
    )
    solver = ALNSSolver(problem, budget, params)
    return solver.solve()
