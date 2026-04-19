"""
Vectorized HGS Algorithm.
"""

from __future__ import annotations

import random
import time
from typing import Any, List, Optional, Tuple, Union

import torch

from logic.src.models.policies.hgs_core.crossover import vectorized_ordered_crossover
from logic.src.models.policies.hgs_core.population import VectorizedPopulation
from logic.src.models.policies.local_search import (
    vectorized_relocate,
    vectorized_swap,
    vectorized_swap_star,
    vectorized_three_opt,
    vectorized_two_opt,
    vectorized_two_opt_star,
)
from logic.src.models.policies.shared import vectorized_linear_split
from logic.src.tracking.viz_mixin import PolicyVizMixin


class VectorizedHGS(PolicyVizMixin):
    """
    Main class for the Vectorized Hybrid Genetic Search (HGS) algorithm.
    Orchestrates the evolution process, including initialization, selection, crossover,
    local search (education), and survivor selection.

    Args:
        dist_matrix (torch.Tensor): Distance matrix.
        wastes (torch.Tensor): Wastes tensor.
        vehicle_capacity (float/torch.Tensor): Vehicle capacity.
        max_iterations (int): Maximum number of iterations for local search.
        time_limit (float): Time limit for the search in seconds.
        generator: Torch generator for reproducibility.
    """

    def __init__(
        self,
        dist_matrix: torch.Tensor,
        wastes: torch.Tensor,
        vehicle_capacity: Any,
        max_iterations: int = 50,
        time_limit: float = 1.0,
        device: str = "cuda",
        seed: int = 42,
        generator: Optional[torch.Generator] = None,
        rng: Optional[random.Random] = None,
    ):
        """
        Initialize the HGS solver.
        """
        self.dist_matrix = dist_matrix
        self.wastes = wastes
        self.vehicle_capacity = vehicle_capacity
        self.max_iterations = max_iterations
        self.time_limit = time_limit
        self.device = torch.device(device)
        self.seed = seed
        if generator is not None:
            self.generator = generator
        else:
            self.generator = torch.Generator(device=self.device).manual_seed(self.seed)

        if rng is not None:
            self.rng = rng
        else:
            self.rng = random.Random(self.seed)

    def __getstate__(self):
        """Prepare state for pickling."""
        state = self.__dict__.copy()
        if "generator" in state and state["generator"] is not None:
            state["generator_state"] = state["generator"].get_state()
            state["generator_device"] = str(state["generator"].device)
            del state["generator"]
        return state

    def __setstate__(self, state):
        """Restore state after unpickling."""
        gen_state = state.pop("generator_state", None)
        gen_device = state.pop("generator_device", None)
        self.__dict__.update(state)
        if gen_state is not None:
            self.generator = torch.Generator(device=gen_device)
            self.generator.set_state(gen_state)
        else:
            self.generator = torch.Generator(device="cpu")

    def solve(
        self,
        initial_solutions: torch.Tensor,
        n_generations: int = 50,
        population_size: int = 10,
        elite_size: int = 5,
        time_limit: Optional[float] = None,
        max_vehicles: int = 0,
        crossover_rate: float = 0.7,
    ) -> Tuple[Union[torch.Tensor, List[List[Any]]], torch.Tensor]:
        """
        Runs the HGS algorithm starting from a set of initial solutions (Expert Imitation Mode).

        Args:
            initial_solutions (torch.Tensor): Initial solutions (giant tours) (B, N).
            n_generations (int): Number of generations to run.
            population_size (int): Size of the genetic population.
            elite_size (int): Number of elite individuals to preserve during restarting.
            time_limit (float, optional): Override time limit in seconds.
            max_vehicles (int, optional): Maximum number of vehicles allowed (0 for unlimited).
            crossover_rate (float): Probability of applying crossover per generation.

        Returns:
            tuple: (best_routes, best_cost)
        """
        B, N = initial_solutions.size()
        start_time = time.time()

        # Initial Evaluation
        _, costs = vectorized_linear_split(
            initial_solutions,
            self.dist_matrix,
            self.wastes,
            self.vehicle_capacity,
            max_vehicles=max_vehicles,
        )

        pop = VectorizedPopulation(population_size, self.device, self.generator)
        # 1. Initialization: Create the initial population from guest solutions (Expert Imitation).
        # In HGS, the population is split into feasible and infeasible sub-populations.
        pop.initialize(initial_solutions, costs, elite_size)

        no_improv = 0
        best_cost_tracker = pop.costs.min().item()

        # 2. Main Evolutionary Loop
        for _gen in range(n_generations):
            if time_limit is not None and (time.time() - start_time > time_limit):
                break

            # 3. Selection: Binary Tournament Selection
            # Select two parents from the population. HGS uses biased fitness which
            # combines objective value and a diversity rank based on distance to neighbors.
            p1, p2 = pop.get_parents(n_offspring=1)
            p1 = p1.squeeze(1)
            p2 = p2.squeeze(1)

            # 4. Crossover: Ordered Crossover (OX) for giant tours
            # Combines segments from parents while maintaining relative order of cities.
            # With probability (1 - crossover_rate), skip crossover and clone parent.
            offspring_giant = (
                vectorized_ordered_crossover(p1, p2, self.device, self.generator)
                if self.rng.random() < crossover_rate
                else p1.clone()
            )

            # 5. Evaluation & Splitting:
            # Convert the giant tour offspring into actual routes using a linear split algorithm.
            routes_list, split_costs = vectorized_linear_split(
                offspring_giant,
                self.dist_matrix,
                self.wastes,
                self.vehicle_capacity,
                max_vehicles=max_vehicles,
            )

            # 6. Education Phase (Local Search Intensification):
            # Apply a sequence of neighborhood operators to improve the offspring's cost.
            # This 'Education' step is what makes it a 'Hybrid' genetic algorithm.
            improved_routes, improved_costs = self.educate(routes_list, split_costs, max_vehicles)

            # 7. Giant Tour Reconstruction:
            # Flatten the improved routes back into a single sequence for the next generation.
            giant_candidates = torch.zeros((B, N), dtype=torch.long, device=self.device)
            improved_routes_cpu = improved_routes.cpu().numpy()

            for b in range(B):
                r_full = improved_routes_cpu[b]
                g_tour = r_full[r_full != 0]
                if len(g_tour) == N:
                    giant_candidates[b, :] = torch.tensor(g_tour, device=self.device)
                else:
                    giant_candidates[b, :] = offspring_giant[b, :]

            # 8. Survivor Selection:
            # Add the improved offspring back into the population.
            # The population manager will then prune individuals to maintain the size limit,
            # using 'Biased Fitness' to balance objective quality and population diversity.
            pop.add_individuals(giant_candidates, improved_costs, elite_size)

            # 9. Stagnation Check & Restart:
            # If the best solution hasn't improved for many generations, trigger a restart.
            current_best = pop.costs.min().item()
            if current_best < best_cost_tracker - 1e-4:
                best_cost_tracker = current_best
                no_improv = 0
            else:
                no_improv += 1

            restarted = False
            if no_improv > 50:
                # Restart Mechanism:
                # Keep the elite individuals and replace the rest with new random solutions.
                # This helps escape local optima and explores different regions of the search space.
                k = elite_size
                # Keep Elite
                sorted_idx = torch.argsort(pop.biased_fitness, dim=1)[:, :k]
                elite_pop = torch.gather(pop.population, 1, sorted_idx.unsqueeze(2).expand(-1, -1, N))
                elite_cost = torch.gather(pop.costs, 1, sorted_idx)

                # Generate and evaluate new random solutions to diversify the population.
                n_rest = pop.max_size - k
                new_pop = torch.zeros((B, n_rest, N), dtype=torch.long, device=self.device)
                for b in range(B):
                    for i in range(n_rest):
                        new_pop[b, i] = torch.randperm(N, device=self.device, generator=self.generator) + 1

                # Evaluate new population...
                b_sz, n_rst, n_nds = new_pop.shape
                flat_pop = new_pop.view(b_sz * n_rst, n_nds)

                flat_dist = self.dist_matrix.repeat_interleave(n_rst, dim=0)
                flat_wastes = self.wastes.repeat_interleave(n_rst, dim=0)
                flat_cap = self.vehicle_capacity
                if isinstance(flat_cap, torch.Tensor) and flat_cap.dim() > 0:
                    flat_cap = flat_cap.repeat_interleave(n_rst, dim=0)

                _, new_costs_flat = vectorized_linear_split(
                    flat_pop,
                    flat_dist,
                    flat_wastes,
                    flat_cap,
                    max_vehicles=max_vehicles,
                )
                new_costs = new_costs_flat.view(b_sz, n_rst)

                # Merge elite and new individuals back into the population.
                pop.population = torch.cat([elite_pop, new_pop], dim=1)
                pop.costs = torch.cat([elite_cost, new_costs], dim=1)
                pop.compute_biased_fitness(elite_size)
                no_improv = 0
                restarted = True

            self._viz_record(
                generation=_gen,
                best_cost=float(pop.costs.min().item()),
                mean_cost=float(pop.costs.mean().item()),
                worst_cost=float(pop.costs.max().item()),
                no_improv=no_improv,
                restarted=restarted,
            )

        best_cost_pop, best_idx = torch.min(pop.costs, dim=1)
        best_giant = pop.population[torch.arange(B), best_idx]
        best_routes, best_final_costs = vectorized_linear_split(
            best_giant,
            self.dist_matrix,
            self.wastes,
            self.vehicle_capacity,
            max_vehicles=max_vehicles,
        )
        return best_routes, best_final_costs

    def educate(
        self, routes_list: list[list[int]], split_costs: torch.Tensor, max_vehicles: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Education Phase (Local Search Intensification).
        Applies standard neighborhood operators.
        """
        B = len(routes_list)
        max_l = max(len(r) for r in routes_list)
        offspring_routes = torch.zeros((B, max_l), dtype=torch.long, device=self.device)
        for b in range(B):
            r = routes_list[b]
            offspring_routes[b, : len(r)] = torch.tensor(r, device=self.device)

        if max_l > 2:
            # Apply various local search moves iteratively
            improved_routes = vectorized_two_opt(
                offspring_routes, self.dist_matrix, max_iterations=self.max_iterations, generator=self.generator
            )
            improved_routes = vectorized_three_opt(
                improved_routes, self.dist_matrix, max_iterations=self.max_iterations, generator=self.generator
            )
            improved_routes = vectorized_swap(
                improved_routes, self.dist_matrix, max_iterations=self.max_iterations, generator=self.generator
            )
            improved_routes = vectorized_relocate(
                improved_routes, self.dist_matrix, max_iterations=self.max_iterations, generator=self.generator
            )
            improved_routes = vectorized_two_opt_star(
                improved_routes, self.dist_matrix, max_iterations=self.max_iterations, generator=self.generator
            )
            improved_routes = vectorized_swap_star(
                improved_routes, self.dist_matrix, max_iterations=self.max_iterations, generator=self.generator
            )

            # Re-calculate costs after local search optimization
            from_n = improved_routes[:, :-1]
            to_n = improved_routes[:, 1:]

            # Flexible distance matrix support (batch, non-batch, 2D, 3D)
            if self.dist_matrix.dim() == 3 and self.dist_matrix.size(0) == B:
                batch_ids = torch.arange(B, device=self.device).view(B, 1)
                dists = self.dist_matrix[batch_ids, from_n, to_n]
            else:
                expanded_dm = self.dist_matrix
                if expanded_dm.dim() == 2:
                    expanded_dm = expanded_dm.unsqueeze(0)
                if expanded_dm.size(0) == 1 and B > 1:
                    expanded_dm = expanded_dm.expand(B, -1, -1)
                batch_ids = torch.arange(B, device=self.device).view(B, 1)
                dists = expanded_dm[batch_ids, from_n, to_n]

            improved_costs = dists.sum(dim=1)
        else:
            improved_routes = offspring_routes
            improved_costs = split_costs

        return improved_routes, improved_costs
