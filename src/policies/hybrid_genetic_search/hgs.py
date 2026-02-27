"""
Hybrid Genetic Search (HGS) policy module.

Combines genetic algorithms with local search and the Split algorithm
for solving the Capacitated Vehicle Routing Problem with Profits.

Reference:
    Vidal, T., Crainic, T. G., Gendreau, M., & Prins, C. (2016). A unified
    solution framework for multi-attribute vehicle routing problems. European
    Journal of Operational Research, 234(3), 658-673.
"""

import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.tracking.viz_mixin import PolicyVizMixin

from ..local_search.local_search_hgs import HGSLocalSearch
from .evolution import evaluate, ordered_crossover, update_biased_fitness
from .individual import Individual
from .params import HGSParams
from .pyvrp_wrapper import solve_pyvrp
from .split import LinearSplit


class HGSSolver(PolicyVizMixin):
    """
    Implements Hybrid Genetic Search for VRPP.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: HGSParams,
        mandatory_nodes: Optional[List[int]] = None,
    ):
        """
        Initialize the HGS solver.

        Args:
            dist_matrix: NxN distance matrix.
            wastes: Dictionary of node wastes.
            capacity: Maximum vehicle capacity.
            R: Revenue multiplier.
            C: Cost multiplier.
            params: Detailed HGS parameters.
            mandatory_nodes: List of local node indices that MUST be visited.
        """
        self.d = dist_matrix
        self.wastes = wastes
        self.Q = capacity
        self.R = R
        self.C = C
        self.params = params
        self.mandatory_nodes = mandatory_nodes

        self.n_nodes = len(dist_matrix) - 1
        self.nodes = list(range(1, self.n_nodes + 1))

        self.split_manager = LinearSplit(dist_matrix, wastes, capacity, R, C, params.max_vehicles, mandatory_nodes)
        self.ls = HGSLocalSearch(dist_matrix, wastes, capacity, R, C, params)

    def solve(self) -> Tuple[List[List[int]], float, float]:
        """
        Run the Hybrid Genetic Search algorithm.

        Returns:
            Tuple[List[List[int]], float, float]: Best routes, total profit, and total cost.
        """
        population: List[Individual] = []

        # 1. Initial Population
        for _ in range(self.params.population_size):
            gt = self.nodes[:]
            random.shuffle(gt)
            ind = Individual(gt)
            evaluate(ind, self.split_manager)
            population.append(ind)

        update_biased_fitness(population, self.params.elite_size)

        start_time = time.time()
        it = 0
        best_profit_so_far = max(ind.profit_score for ind in population)
        while time.time() - start_time < self.params.time_limit:
            it += 1
            # 2. Selection & Crossover
            p1, p2 = self._select_parents(population)
            child = ordered_crossover(p1, p2)

            # 3. Local Search (Mutation)
            if random.random() < self.params.mutation_rate:
                evaluate(child, self.split_manager)
                child = self.ls.optimize(child)

            evaluate(child, self.split_manager)
            population.append(child)

            if child.profit_score > best_profit_so_far:
                best_profit_so_far = child.profit_score

            # 4. Survivor Selection
            if len(population) > self.params.population_size * 2:
                update_biased_fitness(population, self.params.elite_size)
                population.sort(key=lambda x: x.fitness)
                population = population[: self.params.population_size]

            self._viz_record(
                iteration=it,
                best_profit=best_profit_so_far,
                child_profit=child.profit_score,
                child_cost=child.cost,
                population_size=len(population),
            )

        update_biased_fitness(population, self.params.elite_size)
        best_ind = min(population, key=lambda x: -x.profit_score)

        return best_ind.routes, best_ind.profit_score, best_ind.cost

    def _select_parents(self, population: List[Individual]) -> Tuple[Individual, Individual]:
        # Binary Tournament
        def tournament():
            """Perform a binary tournament selection."""
            i1, i2 = random.sample(population, 2)
            return i1 if i1.fitness < i2.fitness else i2

        return tournament(), tournament()


def run_hgs(dist_matrix, wastes, capacity, R, C, values, mandatory_nodes=None, *args):
    """
    Main HGS entry point with dispatching logic.

    Args:
        dist_matrix: Distance matrix.
        wastes: Bin wastes.
        capacity: Vehicle capacity.
        R: Revenue multiplier.
        C: Cost multiplier.
        values: Dictionary of parameters and config.
        mandatory_nodes: List of local node indices that MUST be visited.
        *args: Additional arguments (ignored or passed through).

    Returns:
        Tuple[List[List[int]], float, float]: Best routes, profit, and cost.
    """
    engine = values.get("engine") or values.get("variant")
    if engine == "pyvrp":
        return solve_pyvrp(dist_matrix, wastes, capacity, R, C, values)

    if len(dist_matrix) <= 1:
        return [], 0.0, 0.0

    if len(dist_matrix) == 2:
        d = wastes.get(1, 0)
        if d <= capacity:
            # Calculate simple profit/cost
            cost = dist_matrix[0][1] + dist_matrix[1][0]
            profit = d * R
            return [[1]], profit, C * cost
        else:
            return [], 0.0, 0.0

    params = HGSParams(
        time_limit=values.get("time_limit", 10),
        population_size=values.get("population_size", 50),
        elite_size=values.get("elite_size", 10),
        mutation_rate=values.get("mutation_rate", 0.2),
        n_generations=values.get("n_generations", 100),
        max_vehicles=values.get("max_vehicles", 0),
    )
    solver = HGSSolver(dist_matrix, wastes, capacity, R, C, params, mandatory_nodes)
    return solver.solve()
