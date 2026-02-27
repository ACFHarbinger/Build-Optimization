"""
Particle Swarm Optimization Memetic Algorithm (PSOMA) for VRPP.

Particles navigate the discrete routing space via swap-based velocity.
A low inertia weight (ω≈0.4) forces intensive local exploitation.  A
periodic memetic step applies worst-removal + greedy-insertion local
search to each particle, analogous to the genetic operators described in
the survey.

Reference:
    Survey §"Particle Swarm Optimization" — PSOMA with 0.016% optimality gap.
"""

import contextlib
import copy
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.tracking.viz_mixin import PolicyVizMixin

from ..operators.destroy_operators import worst_removal
from ..operators.repair_operators import greedy_insertion
from .params import PSOMAParams
from .particle import PSOMAParticle


class PSOMAsSolver(PolicyVizMixin):
    """
    PSOMA solver for VRPP — PSO with memetic local-search steps.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: PSOMAParams,
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

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def solve(self) -> Tuple[List[List[int]], float, float]:
        """
        Run PSOMA optimisation.

        Returns:
            Tuple of (routes, profit, cost).
        """
        if self.n_nodes == 0:
            return [], 0.0, 0.0

        start = time.time()

        # Initialise swarm
        swarm = self._init_swarm()
        gbest_routes, gbest_profit = self._global_best(swarm)
        gbest_cost = self._cost(gbest_routes)

        for iteration in range(self.params.max_iterations):
            if time.time() - start > self.params.time_limit:
                break

            for particle in swarm:
                # Velocity / position update via probabilistic segment adoption
                particle.routes = self._update_position(
                    particle.routes,
                    particle.pbest_routes,
                    gbest_routes,
                )
                particle.profit = self._evaluate(particle.routes)

                # Update personal best
                if particle.profit > particle.pbest_profit:
                    particle.pbest_routes = copy.deepcopy(particle.routes)
                    particle.pbest_profit = particle.profit

            # Global best update
            for particle in swarm:
                if particle.profit > gbest_profit:
                    gbest_routes = copy.deepcopy(particle.routes)
                    gbest_profit = particle.profit
                    gbest_cost = self._cost(gbest_routes)

            # Memetic step: periodic local search on every particle
            if (iteration + 1) % self.params.local_search_freq == 0:
                for particle in swarm:
                    ls_routes = self._local_search(particle.routes)
                    ls_profit = self._evaluate(ls_routes)
                    if ls_profit > particle.profit:
                        particle.routes = ls_routes
                        particle.profit = ls_profit
                        if ls_profit > particle.pbest_profit:
                            particle.pbest_routes = copy.deepcopy(ls_routes)
                            particle.pbest_profit = ls_profit
                        if ls_profit > gbest_profit:
                            gbest_routes = copy.deepcopy(ls_routes)
                            gbest_profit = ls_profit
                            gbest_cost = self._cost(gbest_routes)

            self._viz_record(
                iteration=iteration,
                best_profit=gbest_profit,
                best_cost=gbest_cost,
                swarm_size=len(swarm),
            )

        return gbest_routes, gbest_profit, gbest_cost

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _init_swarm(self) -> List[PSOMAParticle]:
        """Initialise swarm with random feasible solutions."""
        swarm = []
        for _ in range(self.params.pop_size):
            routes = self._build_random_solution()
            profit = self._evaluate(routes)
            swarm.append(PSOMAParticle(routes, profit))
        return swarm

    def _build_random_solution(self) -> List[List[int]]:
        """Order-dependent sequential construction (matches ALNS style)."""
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

    def _global_best(self, swarm: List[PSOMAParticle]) -> Tuple[List[List[int]], float]:
        """Return (routes, profit) of best particle."""
        best = max(swarm, key=lambda p: p.profit)
        return copy.deepcopy(best.routes), best.profit

    def _update_position(
        self,
        current: List[List[int]],
        pbest: List[List[int]],
        gbest: List[List[int]],
    ) -> List[List[int]]:
        """
        Update particle position via OX crossover toward pbest and gbest.
        """
        routes = copy.deepcopy(current)

        # Cognitive component: crossover with pbest
        if random.random() < self.params.c1 * random.random() and pbest:
            routes = self._crossover(routes, pbest)

        # Social component: crossover with gbest
        if random.random() < self.params.c2 * random.random() and gbest:
            routes = self._crossover(routes, gbest)

        # Inertia: with prob (1-omega) randomly relocate one node
        if random.random() > self.params.omega:
            routes = self._random_relocate(routes)

        # 2-opt local search after every position update
        from logic.src.policies.local_search.local_search_aco import ACOLocalSearch

        ls = ACOLocalSearch(self.dist_matrix, self.wastes, self.capacity, self.R, self.C, self.params)
        return ls.optimize(routes)

    def _crossover(self, base_routes: List[List[int]], guide_routes: List[List[int]]) -> List[List[int]]:
        """
        OX crossover: inject a random segment from guide into base preserving order.
        """
        winner_flat = [n for r in guide_routes for n in r]
        loser_flat = [n for r in base_routes for n in r]

        if len(winner_flat) < 2:
            return copy.deepcopy(base_routes)

        a = random.randint(0, len(winner_flat) - 1)
        b = random.randint(a, min(a + max(1, len(winner_flat) // 3), len(winner_flat)))
        segment = winner_flat[a:b]
        segment_set = set(segment)

        remaining = [n for n in loser_flat if n not in segment_set]
        insert_pos = min(a, len(remaining))
        child_flat = remaining[:insert_pos] + segment + remaining[insert_pos:]

        child_routes: List[List[int]] = []
        curr_route: List[int] = []
        load = 0.0
        for node in child_flat:
            waste = self.wastes.get(node, 0.0)
            if load + waste <= self.capacity:
                curr_route.append(node)
                load += waste
            else:
                if curr_route:
                    child_routes.append(curr_route)
                curr_route = [node]
                load = waste
        if curr_route:
            child_routes.append(curr_route)

        visited = {n for r in child_routes for n in r}
        for n in self.mandatory_nodes:
            if n not in visited:
                child_routes.append([n])

        return child_routes

    def _random_relocate(self, routes: List[List[int]]) -> List[List[int]]:
        """
        Randomly remove one node and reinsert it at a random position.

        Args:
            routes: Current routes.

        Returns:
            Perturbed routes.
        """
        flat = [n for r in routes for n in r]
        if not flat:
            return routes
        node = random.choice(flat)
        new_routes = [[n for n in r if n != node] for r in routes]
        new_routes = [r for r in new_routes if r]
        with contextlib.suppress(Exception):
            new_routes = greedy_insertion(
                new_routes,
                [node],
                self.dist_matrix,
                self.wastes,
                self.capacity,
                R=self.R,
                mandatory_nodes=self.mandatory_nodes,
            )
        return new_routes

    def _local_search(self, routes: List[List[int]]) -> List[List[int]]:
        """
        Memetic local search: worst-removal + greedy-insertion + ACO.
        """
        n = max(3, self.params.n_removal)
        try:
            partial, removed = worst_removal(routes, n, self.dist_matrix)
            repaired = greedy_insertion(
                partial,
                removed,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                R=self.R,
                mandatory_nodes=self.mandatory_nodes,
            )
            from logic.src.policies.local_search.local_search_aco import ACOLocalSearch

            ls = ACOLocalSearch(self.dist_matrix, self.wastes, self.capacity, self.R, self.C, self.params)
            return ls.optimize(repaired)
        except Exception:
            return copy.deepcopy(routes)

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
