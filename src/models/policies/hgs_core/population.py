"""
HGS Population Management.
"""

from typing import Any, Optional, Tuple

import torch

from .evaluation import calc_broken_pairs_distance


class VectorizedPopulation:
    """
    Manages a population of solutions for the HGS algorithm.

    Handles:
    - Storage of solutions (giant tours), costs, and fitness metrics.
    - Diversity calculation (Broken Pairs Distance).
    - Biased fitness computation (ranking by cost and diversity).
    - Survivor selection.

    Args:
        size (int): Maximum population size.
        device: Torch device.
        generator (Optional[torch.Generator]): Torch random number generator.
    """

    def __init__(self, size: int, device: Any, generator: Optional[torch.Generator] = None):
        """
        Initialize the population.

        Args:
            size (int): Max population size.
            device: Computing device.
            generator (Optional[torch.Generator]): Torch random number generator.
        """
        self.max_size = size
        self.device = device
        self.generator = generator
        self.population: torch.Tensor = torch.empty(0)  # (B, P, N)
        self.costs: torch.Tensor = torch.empty(0)  # (B, P)
        self.biased_fitness: torch.Tensor = torch.empty(0)  # (B, P)
        self.diversity_scores: torch.Tensor = torch.empty(0)  # (B, P)

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

    def initialize(self, initial_pop: torch.Tensor, initial_costs: torch.Tensor, nb_elite: int):
        """
        Initializes the population with a set of starting solutions.

        Args:
            initial_pop (torch.Tensor): Initial solutions (giant tours). Shape (B, N) or (B, P0, N).
            initial_costs (torch.Tensor): Costs of initial solutions. Shape (B,) or (B, P0).
            nb_elite (int): Number of elite individuals for biased fitness.
        """

        if initial_pop.dim() == 2:
            initial_pop = initial_pop.unsqueeze(1)  # (B, 1, N)

        # Consistent costs shape (B, P0)
        if initial_costs.dim() == 1:
            initial_costs = initial_costs.unsqueeze(1)  # (B, 1)

        self.population = initial_pop
        self.costs = initial_costs
        self.compute_biased_fitness(nb_elite)

    def add_individuals(self, candidates: torch.Tensor, costs: torch.Tensor, nb_elite: int):
        """
        Merges new individuals into the population and selects survivors based on biased fitness.
        Maintains the population size at or below `max_size`.

        Args:
            candidates (torch.Tensor): New solutions to add. Shape (B, C, N).
            costs (torch.Tensor): Costs of new solutions. Shape (B, C).
            nb_elite (int): Number of elite individuals for biased fitness.
        """

        if candidates.dim() == 2:
            candidates = candidates.unsqueeze(1)
            costs = costs.unsqueeze(1)

        # 1. Concatenate
        combined_pop = torch.cat([self.population, candidates], dim=1)  # (B, P+C, N)
        combined_costs = torch.cat([self.costs, costs], dim=1)  # (B, P+C)

        # 2. Update state temporarily
        self.population = combined_pop
        self.costs = combined_costs

        # 3. Compute Fitness & Survivor Selection
        self.compute_biased_fitness(nb_elite)

        # Select best P based on biased fitness
        # We want smallest biased_fitness (rank sum)

        if self.population.size(1) > self.max_size:
            # Sort by fitness (ascending, smaller is better)
            _, indices = torch.sort(self.biased_fitness, dim=1)
            survivors = indices[:, : self.max_size]  # (B, max_size)

            # Gather survivors
            B, _, N = self.population.size()

            # Gather population
            surv_expanded = survivors.unsqueeze(2).expand(-1, -1, N)
            self.population = torch.gather(self.population, 1, surv_expanded)

            # Gather costs
            self.costs = torch.gather(self.costs, 1, survivors)

            # Gather fitness
            self.biased_fitness = torch.gather(self.biased_fitness, 1, survivors)

            # Gather diversity (optional)
            if self.diversity_scores is not None and self.diversity_scores.numel() > 0:
                self.diversity_scores = torch.gather(self.diversity_scores, 1, survivors)

    def compute_biased_fitness(self, nb_elite: int):
        """
        Compute Biased Fitness for all individuals in the population.
        BF(I) = Rank_C(I) + (1 - N_elite/|Pop|) * Rank_D(I).
        Lower is better for both ranks (0 is best).
        Updates `self.biased_fitness` and `self.diversity_scores`.

        Args:
            nb_elite (int): Number of elite individuals to protect.
        """

        B, P, N = self.population.size()

        # 1. Rank by Cost (Ascending: lower cost is better, Rank 0)
        cost_indices = torch.argsort(self.costs, dim=1)
        cost_ranks = torch.argsort(cost_indices, dim=1).float()

        # 2. Diversity
        # Calculate diversity (higher is better)
        self.diversity_scores = calc_broken_pairs_distance(self.population)

        # Rank by Diversity (Descending: higher diversity is better, Rank 0)
        div_indices = torch.argsort(self.diversity_scores, dim=1, descending=True)
        div_ranks = torch.argsort(div_indices, dim=1).float()

        # 3. Combine
        # Parameterless Biased Fitness (Vidal 2022)
        diversity_weight = 1.0 - (nb_elite / P)
        self.biased_fitness = cost_ranks + diversity_weight * div_ranks

    def get_parents(self, n_offspring: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Selects parents for crossover using binary tournament selection based on biased fitness.

        Args:
            n_offspring (int): Number of offspring to produce (pairs of parents).

        Returns:
            tuple: (parents1, parents2). Both valid tensors of shape (B, n_offspring, N).
        """

        B, P, N = self.population.size()

        def tournament():
            """Selects indices using binary tournament."""
            idx_a = torch.randint(0, P, (B, n_offspring), device=self.device, generator=self.generator)
            idx_b = torch.randint(0, P, (B, n_offspring), device=self.device, generator=self.generator)
            fit_a = torch.gather(self.biased_fitness, 1, idx_a)
            fit_b = torch.gather(self.biased_fitness, 1, idx_b)
            return torch.where(fit_a < fit_b, idx_a, idx_b)

        parent1_idx = tournament()
        parent2_idx = tournament()

        p1 = torch.gather(self.population, 1, parent1_idx.unsqueeze(2).expand(-1, -1, N))
        p2 = torch.gather(self.population, 1, parent2_idx.unsqueeze(2).expand(-1, -1, N))

        return p1, p2
