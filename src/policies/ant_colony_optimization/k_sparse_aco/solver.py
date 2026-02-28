"""
K-Sparse ACO Solver Module for Build Optimization.

Implements ACS with sparse pheromone storage for memory efficiency.
"""

import time
from typing import Optional, Tuple

import numpy as np

from core.problem import BuildProblem
from tracking.viz_mixin import PolicyVizMixin

from .construction import SolutionConstructor
from .params import ACOParams
from .pheromones import SparsePheromoneTau


class KSparseACOSolver(PolicyVizMixin):
    """
    K-Sparse Ant Colony System solver for Build Optimization.
    """

    def __init__(
        self,
        problem: BuildProblem,
        budget: float,
        params: ACOParams,
        seed: Optional[int] = None,
    ):
        """
        Initialize the K-Sparse ACO solver.
        """
        self.problem = problem
        self.budget = budget
        self.params = params

        num_slots = problem.num_slots

        # Heuristic values (eta) based on item cost efficiency
        # Avoid division by zero
        self.eta = 1.0 / (self.problem.costs + 1e-6)

        # Compute initial pheromone
        # In Build Optimization, we'll start with a conservative default tau_0
        if params.tau_0 is None:
            self.tau_0 = (
                1.0 / (num_slots * np.mean(self.problem.costs)) if np.mean(self.problem.costs) > 0 else params.tau_max
            )
        else:
            self.tau_0 = params.tau_0

        # Initialize sparse pheromone matrix
        # (nodes mapped to slots here, so n_nodes = num_slots)
        self.pheromone = SparsePheromoneTau(
            num_slots,
            params.k_sparse,
            self.tau_0,
            params.tau_min,
            params.tau_max,
        )

        # Initialize Constructor
        self.constructor = SolutionConstructor(
            problem,
            budget,
            self.pheromone,
            self.eta,
            params,
            self.tau_0,
        )

    def solve(self) -> Tuple[Optional[np.ndarray], float]:
        """
        Run the K-Sparse ACO algorithm.

        Returns:
            Tuple[np.ndarray, float]: (best_build, best_score)
        """
        best_build: Optional[np.ndarray] = None
        best_score = -float("inf")
        start_time = time.time()

        for _iteration in range(self.params.max_iterations):
            if time.time() - start_time > self.params.time_limit:
                break

            iteration_best_build: Optional[np.ndarray] = None
            iteration_best_score = -float("inf")

            # Each ant constructs a solution
            for _ in range(self.params.n_ants):
                # Use delegated constructor
                build = self.constructor.construct()

                # We could add an item-swap Local Search here
                # if self.params.local_search is True

                if not self.problem.is_feasible(build):
                    continue

                score = self.problem.evaluate(build)

                if score > iteration_best_score:
                    iteration_best_score = score
                    iteration_best_build = build

            # Update global best
            if iteration_best_score > best_score:
                best_score = iteration_best_score
                best_build = iteration_best_build

            # Global pheromone update
            self._global_pheromone_update(best_build, best_score)

            _tau_vals = [v for nbrs in self.pheromone._pheromone.values() for v in nbrs.values()]
            self._viz_record(
                iteration=_iteration,
                best_cost=-best_score,  # viz record assumes cost min, we maximize score
                iter_best_cost=-iteration_best_score,
                tau_mean=float(sum(_tau_vals) / len(_tau_vals)) if _tau_vals else self.pheromone.tau_0,
                tau_max=float(max(_tau_vals)) if _tau_vals else self.pheromone.tau_0,
            )

        return best_build, best_score

    def _global_pheromone_update(self, best_build: Optional[np.ndarray], best_score: float) -> None:
        """
        Apply ACS global pheromone update on best-so-far solution.
        """
        if best_build is None or best_score <= 0:
            return

        # Evaporate all pheromones
        self.pheromone.evaporate_all(self.params.rho)

        # Deposit on best solution edges
        delta = self.params.elitist_weight * best_score

        for slot_idx, item_idx in enumerate(best_build):
            if item_idx != -1:
                self.pheromone.update_edge(slot_idx, item_idx, delta, evaporate=False)
