"""
Hybrid Genetic Search (HGS) policy module.

Combines genetic algorithms with local search and the Split algorithm
for solving the Capacitated Vehicle Routing Problem with Profits.

Reference:
    Vidal, T. (2022). Hybrid genetic search for the CVRP: Open-source
    implementation and SWAP* neighborhood. Computers & Operations Research,
    140, 105643.
"""

import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.policies.helpers.local_search.local_search_hgs import HGSLocalSearch
from logic.src.policies.helpers.operators.crossover_recombination import ordered_crossover

from .evolution import evaluate, update_biased_fitness
from .individual import Individual
from .params import HGSParams
from .split import LinearSplit


class HGSSolver:
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
        x_coords: Optional[np.ndarray] = None,
        y_coords: Optional[np.ndarray] = None,
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
            x_coords: Optional array of x-coordinates indexed by node id (index 0 = depot).
                      When provided together with y_coords, enables SWAP* polar sector pruning.
            y_coords: Optional array of y-coordinates indexed by node id (index 0 = depot).
                      When provided together with x_coords, enables SWAP* polar sector pruning.
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

        self.split_manager = LinearSplit(
            dist_matrix, wastes, capacity, R, C, params.max_vehicles, mandatory_nodes, vrpp=params.vrpp
        )
        self.ls = HGSLocalSearch(dist_matrix, wastes, capacity, R, C, params)
        if x_coords is not None and y_coords is not None:
            self.ls.x_coords = x_coords
            self.ls.y_coords = y_coords

    def _initialize_population(self, penalty_capacity: float) -> Tuple[List[Individual], List[Individual]]:
        """
        Initialize dual subpopulations with random solutions improved by local search.

        Args:
            penalty_capacity: Initial penalty coefficient for capacity violations.

        Returns:
            Tuple of (feasible population, infeasible population).
        """
        pop_feasible: List[Individual] = []
        pop_infeasible: List[Individual] = []

        initial_size = 4 * self.params.mu
        for _ in range(initial_size):
            gt = self.nodes[:]
            self.random.shuffle(gt)
            ind = Individual(gt)
            evaluate(ind, self.split_manager, penalty_capacity)

            # Educate with local search
            ind = self.ls.optimize(ind)
            evaluate(ind, self.split_manager, penalty_capacity)

            # Insert into appropriate subpopulation
            if ind.is_feasible:
                pop_feasible.append(ind)
            else:
                pop_infeasible.append(ind)

        # Initialize fitness for both subpopulations
        update_biased_fitness(pop_feasible, self.params.nb_elite, self.params.nb_close)
        update_biased_fitness(pop_infeasible, self.params.nb_elite, self.params.nb_close)

        return pop_feasible, pop_infeasible

    def _trim_one(self, pop: List[Individual], distance_cache: Optional[dict] = None) -> None:
        """Trim one sub-population to mu, removing clones first then worst by fitness."""
        # Fix 1: Subpopulation survivor selection is triggered when size reaches mu + lambda
        if len(pop) < self.params.mu + self.params.n_offspring:
            return

        # Fix 5: Recompute fitness after each individual is removed so ranks remain valid
        while len(pop) > self.params.mu:
            update_biased_fitness(pop, self.params.nb_elite, self.params.nb_close, distance_cache)
            clone_idx = self._find_clone(pop)
            if clone_idx is not None:
                pop.pop(clone_idx)
                continue
            worst_idx = max(range(len(pop)), key=lambda i: pop[i].fitness)
            pop.pop(worst_idx)

    def _find_clone(self, pop: List[Individual]) -> Optional[int]:
        """Return the index of any individual whose routes are identical to another's, or None."""
        seen: set = set()
        for i, ind in enumerate(pop):
            # Use a frozenset of tuples as a hashable route signature
            key = frozenset(tuple(r) for r in ind.routes)
            if key in seen:
                return i
            seen.add(key)
        return None

    def _get_diversity_distance(
        self,
        ind_a: Individual,
        ind_b: Individual,
        cache: dict,
    ) -> float:
        """Get diversity distance between two individuals using object-ID based cache."""
        from .evolution import _compute_broken_pairs_distance

        key = (id(ind_a), id(ind_b))
        if key not in cache:
            d = _compute_broken_pairs_distance(ind_a, ind_b)
            cache[key] = d
            cache[(id(ind_b), id(ind_a))] = d
        return cache[key]

    def solve(self) -> Tuple[List[List[int]], float, float]:
        """
        Run the Hybrid Genetic Search algorithm following Vidal et al. (2022).

        Returns:
            Tuple[List[List[int]], float, float]: Best routes, total profit, and total cost.
        """
        start_time = time.process_time()

        # Initialize dual subpopulations and penalty
        penalty_capacity = self.params.initial_penalty_capacity
        if self.params.acceptance_criterion is not None:
            self.params.acceptance_criterion.setup(0.0)  # Placeholder or actual best profit
        pop_feasible, pop_infeasible = self._initialize_population(penalty_capacity)

        # Fix 4: best_profit_so_far initialized from feasible pool only.
        # Track the best feasible individual across all restarts (Fix 3).
        best_feasible_ind: Optional[Individual] = (
            max(pop_feasible, key=lambda x: x.profit_score) if pop_feasible else None
        )
        best_profit_so_far = best_feasible_ind.profit_score if best_feasible_ind is not None else -float("inf")

        it = 0
        it_no_improvement = 0
        diversity_cache: dict = {}
        use_time_limit = self.params.time_limit > 0
        restart_threshold = (
            self.params.restart_timer if self.params.restart_timer > 0 else self.params.n_iterations_no_improvement
        )

        # Main evolutionary loop (Algorithm 1)
        while True:
            # --- Primary stopping criterion ---
            elapsed = time.process_time() - start_time
            if use_time_limit and elapsed >= self.params.time_limit:
                break
            if not use_time_limit and it_no_improvement >= self.params.n_iterations_no_improvement:
                break

            # --- Restart logic (only when time_limit > 0) ---
            if use_time_limit and it_no_improvement >= restart_threshold:
                penalty_capacity = self.params.initial_penalty_capacity
                pop_feasible, pop_infeasible = self._initialize_population(penalty_capacity)
                it_no_improvement = 0
                diversity_cache.clear()  # Fix 9: Invalidate cache after re-initialization

            it += 1
            it_no_improvement += 1

            # Generate and educate offspring
            combined = pop_feasible + pop_infeasible
            if len(combined) < 2:
                break

            child = self._generate_offspring(pop_feasible, pop_infeasible, penalty_capacity, diversity_cache)

            # Insert into subpopulation and optionally repair
            self._insert_and_repair(child, pop_feasible, pop_infeasible, penalty_capacity)

            # Track improvement in feasible best
            if child.is_feasible and child.profit_score > best_profit_so_far:
                best_profit_so_far = child.profit_score
                best_feasible_ind = child
                it_no_improvement = 0

            # Fix 1 & 5: Trim populations conditionally after insertion
            if len(pop_feasible) >= self.params.mu + self.params.n_offspring:
                self._trim_one(pop_feasible)
                diversity_cache.clear()  # Fix 9: Invalidate cache after trim
            if len(pop_infeasible) >= self.params.mu + self.params.n_offspring:
                self._trim_one(pop_infeasible)
                diversity_cache.clear()

            # Step the criterion after offspring education (Algorithm state update)
            if self.params.acceptance_criterion is not None:
                self.params.acceptance_criterion.step(
                    current_obj=best_profit_so_far,
                    candidate_obj=child.profit_score if child.is_feasible else -float("inf"),
                    accepted=child.is_feasible and child.profit_score > best_profit_so_far,
                    iteration=it,
                    max_iterations=self.params.n_iterations_no_improvement,
                )

            # Adjust penalties (batched every 100 iterations per Vidal 2022)
            if it % 100 == 0:
                penalty_capacity = self._adjust_penalties(pop_feasible, pop_infeasible, penalty_capacity)

            getattr(self, "_viz_record", lambda **k: None)(
                iteration=it,
                best_profit=best_profit_so_far,
                child_profit=child.profit_score,
                child_cost=child.cost,
                population_size=len(pop_feasible) + len(pop_infeasible),
            )

        # Return global best feasible found across all restarts
        if best_feasible_ind is not None:
            return (
                best_feasible_ind.routes,
                best_feasible_ind.profit_score,
                best_feasible_ind.cost,
            )
        return self._get_best_solution(pop_feasible, pop_infeasible)

    def _generate_offspring(
        self,
        pop_feasible: List[Individual],
        pop_infeasible: List[Individual],
        penalty_capacity: float,
        diversity_cache: Optional[dict] = None,
    ) -> Individual:
        """Generate and educate offspring via crossover and local search."""
        # Fix 2: Ranks are maintained per subpopulation as specified in Vidal (2022).
        # Recompute before selection so tournament uses current values.
        if pop_feasible:
            update_biased_fitness(pop_feasible, self.params.nb_elite, self.params.nb_close, diversity_cache)
        if pop_infeasible:
            update_biased_fitness(pop_infeasible, self.params.nb_elite, self.params.nb_close, diversity_cache)

        combined = pop_feasible + pop_infeasible
        p1, p2 = self._select_parents(combined)
        child = ordered_crossover(p1, p2, rng=self.random)
        evaluate(child, self.split_manager, penalty_capacity)
        child = self.ls.optimize(child)
        evaluate(child, self.split_manager, penalty_capacity)
        return child

    def _insert_and_repair(
        self,
        child: Individual,
        pop_feasible: List[Individual],
        pop_infeasible: List[Individual],
        penalty_capacity: float,
    ) -> None:
        """Insert child into subpopulation and optionally repair if infeasible."""
        if child.is_feasible:
            pop_feasible.append(child)
        else:
            pop_infeasible.append(child)
            # Repair with 50% probability
            if self.random.random() < self.params.repair_probability:
                repaired = Individual(child.giant_tour[:])
                evaluate(repaired, self.split_manager, penalty_capacity * 10.0)
                repaired = self.ls.optimize(repaired)
                evaluate(repaired, self.split_manager, penalty_capacity)
                if repaired.is_feasible:
                    pop_feasible.append(repaired)

    def _get_best_solution(
        self, pop_feasible: List[Individual], pop_infeasible: List[Individual]
    ) -> Tuple[List[List[int]], float, float]:
        """Return best feasible solution or best infeasible if none exists."""
        if pop_feasible:
            best_ind = max(pop_feasible, key=lambda x: x.profit_score)
        elif pop_infeasible:
            best_ind = max(pop_infeasible, key=lambda x: x.profit_score)
        else:
            return [], 0.0, 0.0
        return best_ind.routes, best_ind.profit_score, best_ind.cost

    def _adjust_penalties(
        self, pop_feasible: List[Individual], pop_infeasible: List[Individual], current_penalty: float
    ) -> float:
        """
        Adjust penalty coefficients to maintain target proportion of feasible solutions.
        Following Vidal et al. (2022), targets ~20% feasible solutions.
        Called every 100 iterations as specified in the paper.

        Args:
            pop_feasible: Feasible subpopulation.
            pop_infeasible: Infeasible subpopulation.
            current_penalty: Current penalty coefficient.

        Returns:
            Updated penalty coefficient.
        """
        total_pop = len(pop_feasible) + len(pop_infeasible)
        if total_pop == 0:
            return current_penalty

        feasible_ratio = len(pop_feasible) / total_pop

        # If too many feasible solutions, decrease penalty (make it easier to violate)
        if feasible_ratio > self.params.target_feasible + 0.05:
            return current_penalty * self.params.penalty_decrease
        # If too many infeasible solutions, increase penalty
        elif feasible_ratio < self.params.target_feasible - 0.05:
            return current_penalty * self.params.penalty_increase
        else:
            return current_penalty

    def _select_parents(self, population: List[Individual]) -> Tuple[Individual, Individual]:
        """Binary tournament selection on pre-computed fitness values."""

        def tournament() -> Individual:
            i1, i2 = self.random.sample(population, 2)
            return i1 if i1.fitness < i2.fitness else i2

        return tournament(), tournament()
