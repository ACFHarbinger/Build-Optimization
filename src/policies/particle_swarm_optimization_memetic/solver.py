"""
Particle Swarm Optimization Memetic Algorithm (PSOMA) for Build Optimization.

Particles navigate the discrete item-slot space via probabilistic updates
toward personal and global bests. A memetic step applies periodic
local search to improve solution quality.
"""

import random
import time
from typing import List, Tuple

import numpy as np

from core.problem import BuildProblem
from tracking.viz_mixin import PolicyVizMixin

from ..operators.destroy_operators import random_removal
from ..operators.repair_operators import greedy_insertion
from .params import PSOMAParams
from .particle import PSOMAParticle


class PSOMAsSolver(PolicyVizMixin):
    """
    PSOMA solver for Build Optimization — PSO with memetic local-search steps.
    """

    def __init__(
        self,
        problem: BuildProblem,
        budget: float,
        params: PSOMAParams,
    ):
        self.problem = problem
        self.budget = budget
        self.params = params

    def solve(self) -> Tuple[np.ndarray, float]:
        """
        Run PSOMA optimisation.

        Returns:
            Tuple of (best_build, best_score).
        """
        start = time.time()

        # Initialise swarm
        swarm = self._init_swarm()
        gbest_build, gbest_score = self._global_best(swarm)

        for iteration in range(self.params.max_iterations):
            if time.time() - start > self.params.time_limit:
                break

            for particle in swarm:
                # Velocity / position update via probabilistic slot adoption
                particle.build = self._update_position(
                    particle.build,
                    particle.pbest_build,
                    gbest_build,
                )

                # Ensure feasibility
                if not self.problem.is_feasible(particle.build):
                    particle.build = self.problem.random_solution()

                particle.score = self.problem.evaluate(particle.build)

                # Update personal best
                if particle.score > particle.pbest_score:
                    particle.pbest_build = particle.build.copy()
                    particle.pbest_score = particle.score

            # Global best update
            for particle in swarm:
                if particle.score > gbest_score:
                    gbest_build = particle.build.copy()
                    gbest_score = particle.score

            # Memetic step: periodic local search on every particle
            if (iteration + 1) % self.params.local_search_freq == 0:
                for particle in swarm:
                    ls_build = self._local_search(particle.build)
                    ls_score = self.problem.evaluate(ls_build)
                    if ls_score > particle.score:
                        particle.build = ls_build
                        particle.score = ls_score
                        if ls_score > particle.pbest_score:
                            particle.pbest_build = ls_build.copy()
                            particle.pbest_score = ls_score
                        if ls_score > gbest_score:
                            gbest_build = ls_build.copy()
                            gbest_score = ls_score

            self._viz_record(
                iteration=iteration,
                best_profit=gbest_score,  # legacy name
                best_cost=-gbest_score,
                swarm_size=len(swarm),
            )

        return gbest_build, gbest_score

    def _init_swarm(self) -> List[PSOMAParticle]:
        """Initialise swarm with random feasible solutions."""
        swarm = []
        for _ in range(self.params.pop_size):
            build = self.problem.random_solution()
            score = self.problem.evaluate(build)
            swarm.append(PSOMAParticle(build, score))
        return swarm

    def _global_best(self, swarm: List[PSOMAParticle]) -> Tuple[np.ndarray, float]:
        """Return (build, score) of best particle."""
        best = max(swarm, key=lambda p: p.score)
        return best.build.copy(), best.score

    def _update_position(
        self,
        current: np.ndarray,
        pbest: np.ndarray,
        gbest: np.ndarray,
    ) -> np.ndarray:
        """
        Update particle position by moving toward pbest and gbest.
        For discrete builds, we adopt values from pbest or gbest with some probability.
        """
        new_build = current.copy()
        for slot in range(self.problem.num_slots):
            r = random.random()
            # Cognitive attraction
            if r < self.params.c1 * random.random():
                new_build[slot] = pbest[slot]
            # Social attraction
            elif r < self.params.c2 * random.random():
                new_build[slot] = gbest[slot]
            # Probabilistic decay / inertia is implicitly handled by not changing slot if r is high

        return new_build

    def _local_search(self, build: np.ndarray) -> np.ndarray:
        """
        Memetic local search: random-removal + greedy-insertion.
        """
        n = self.params.n_removal
        partial = random_removal(build, n, self.problem)
        repaired = greedy_insertion(partial, self.budget, self.problem)
        return repaired
