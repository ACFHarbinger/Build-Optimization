"""
Hybrid Genetic Search with Ruin-and-Recreate (HGS-RR) implementation.

Combines evolutionary search with adaptive Large Neighborhood Search operators
for solving the Capacitated Vehicle Routing Problem with Profits.

Reference:
    Simensen, M., Hasle, G., & Stalhane, M. "Combining hybrid
    genetic search with ruin-and-recreate for solving the
    capacitated vehicle routing problem", 2022
"""

import random
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from logic.src.policies.helpers.operators.crossover_recombination import ordered_crossover
from logic.src.policies.route_construction.meta_heuristics.hybrid_genetic_search import Individual
from logic.src.policies.route_construction.meta_heuristics.hybrid_genetic_search.evolution import (
    evaluate,
    update_biased_fitness,
)
from logic.src.policies.route_construction.meta_heuristics.hybrid_genetic_search.split import LinearSplit

from .ruin_recreate import AdaptiveOperatorManager, RuinRecreateOperator


class HGSRRSolver:
    """
    Hybrid Genetic Search with Ruin-and-Recreate for VRPP.

    This algorithm extends standard HGS by replacing traditional local search
    with adaptive destroy/repair operators (ruin-and-recreate).
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: Any,
        mandatory_nodes: Optional[List[int]] = None,
    ):
        """
        Initialize the HGS-RR solver.

        Args:
            dist_matrix: NxN distance matrix.
            wastes: Dictionary of node wastes.
            capacity: Maximum vehicle capacity.
            R: Revenue multiplier.
            C: Cost multiplier.
            params: HGS-RR parameters.
            mandatory_nodes: List of local node indices that MUST be visited.
            seed: Random seed for reproducibility.
        """
        self.d = dist_matrix
        self.wastes = wastes
        self.Q = capacity
        self.R = R
        self.C = C
        self.params = params
        self.mandatory_nodes = mandatory_nodes
        self.random = random.Random(params.seed) if params.seed is not None else random.Random()

        self.n_nodes = len(dist_matrix) - 1
        self.nodes = list(range(1, self.n_nodes + 1))

        # Split manager for evaluating giant tours
        self.split_manager = LinearSplit(
            dist_matrix, wastes, capacity, R, C, params.max_vehicles, mandatory_nodes, params.vrpp
        )

        # Ruin-and-recreate operator with adaptive selection
        self.rr_operator = RuinRecreateOperator(
            dist_matrix=dist_matrix,
            wastes=wastes,
            capacity=capacity,
            revenue=R,
            cost_unit=C,
            params=params,
            split_manager=self.split_manager,
        )

        # Adaptive operator manager
        self.operator_manager = AdaptiveOperatorManager(
            destroy_operators=params.destroy_operators,
            repair_operators=params.repair_operators,
            reaction_factor=params.reaction_factor,
            decay_parameter=params.decay_parameter,
        )

    def solve(self) -> Tuple[List[List[int]], float, float]:
        """
        Run the HGS-RR algorithm.

        Returns:
            Tuple[List[List[int]], float, float]: Best routes, total profit, and total cost.
        """
        start_time = time.process_time()
        population: List[Individual] = []

        # 1. Initial Population
        for _ in range(self.params.population_size):
            gt = self.nodes[:]
            self.random.shuffle(gt)
            ind = Individual(gt, expand_pool=self.params.vrpp)
            evaluate(ind, self.split_manager)
            population.append(ind)

        # Note: alpha_diversity parameter removed in Vidal 2022 refactoring
        # Diversity weight is now automatically calculated as: 1 - (N_elite / |Pop|)
        update_biased_fitness(population, self.params.elite_size, self.params.neighbor_list_size)

        it = 0
        it_no_improvement = 0
        best_profit_so_far = max(ind.profit_score for ind in population)
        best_cost_so_far = min(ind.cost for ind in population if ind.profit_score == best_profit_so_far)

        while it_no_improvement < self.params.n_iterations_no_improvement:
            if self.params.time_limit > 0 and time.process_time() - start_time > self.params.time_limit:
                break
            it += 1
            it_no_improvement += 1

            # 2. Selection & Crossover
            p1, p2 = self._select_parents(population)
            # Skip crossover if giant tours are too small (need at least 2 nodes)
            if len(p1.giant_tour) >= 2 and len(p2.giant_tour) >= 2:
                child = ordered_crossover(p1, p2, rng=self.random)
            else:
                # If tours are too small, just copy one parent
                child = Individual(p1.giant_tour[:], expand_pool=self.params.vrpp)

            # 3. Adaptive Ruin-and-Recreate (Mutation)
            if self.random.random() < self.params.mutation_rate:
                # Select operators adaptively
                destroy_op, repair_op = self.operator_manager.select_operators(self.random)
                child_improved = self.rr_operator.apply(
                    individual=child,
                    destroy_operator=destroy_op,
                    repair_operator=repair_op,
                )

                # Score the operators based on improvement
                score = self._compute_operator_score(child_improved, best_profit_so_far, best_cost_so_far, p1, p2)
                self.operator_manager.update_scores(destroy_op, repair_op, score)

                child = child_improved

            # Re-evaluate child
            evaluate(child, self.split_manager)
            child.expand_pool = self.params.vrpp
            population.append(child)

            # Track improvements
            if child.profit_score > best_profit_so_far:
                best_profit_so_far = child.profit_score
                best_cost_so_far = child.cost
                it_no_improvement = 0
            elif child.profit_score == best_profit_so_far and child.cost < best_cost_so_far:
                best_cost_so_far = child.cost
                it_no_improvement = 0

            # Note: Adaptive alpha_diversity removed in Vidal 2022 refactoring
            # Diversity weight now automatically calculated as: 1 - (N_elite / |Pop|)
            # This parameterless approach eliminates the need for manual diversity tuning

            # 4. Survivor Selection
            if len(population) > self.params.population_size * self.params.survivor_threshold:
                update_biased_fitness(population, self.params.elite_size, self.params.neighbor_list_size)
                population.sort(key=lambda x: x.fitness)
                population = population[: self.params.population_size]

            # Decay operator weights periodically
            if it % 10 == 0:
                self.operator_manager.decay_weights()

            getattr(self, "_viz_record", lambda **k: None)(
                iteration=it,
                best_profit=best_profit_so_far,
                child_profit=child.profit_score,
                child_cost=child.cost,
                population_size=len(population),
                operator_entropy=self.operator_manager.entropy(),
            )

        # Final evaluation and selection
        update_biased_fitness(population, self.params.elite_size, self.params.neighbor_list_size)
        best_ind = min(population, key=lambda x: -x.profit_score)

        return best_ind.routes, best_ind.profit_score, best_ind.cost

    def _select_parents(self, population: List[Individual]) -> Tuple[Individual, Individual]:
        """Select two parents using binary tournament selection."""

        def tournament():
            """Perform a binary tournament selection."""
            i1, i2 = self.random.sample(population, 2)
            return i1 if i1.fitness < i2.fitness else i2

        return tournament(), tournament()

    def _compute_operator_score(
        self,
        child: Individual,
        best_profit: float,
        best_cost: float,
        p1: Individual,
        p2: Individual,
    ) -> float:
        """
        Compute adaptive operator score based on solution quality improvement.

        Args:
            child: The offspring individual.
            best_profit: Current best profit in population.
            best_cost: Current best cost for the best profit.
            p1: First parent.
            p2: Second parent.

        Returns:
            Score value (sigma_1, sigma_2, sigma_3, or 0).
        """
        # New global best
        if child.profit_score > best_profit or (child.profit_score == best_profit and child.cost < best_cost):
            return self.params.score_sigma_1

        # Better than either parent
        parent_best_profit = max(p1.profit_score, p2.profit_score)
        if child.profit_score > parent_best_profit:
            return self.params.score_sigma_2

        # Accepted (added to population)
        return self.params.score_sigma_3
