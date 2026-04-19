"""
Vectorized Augmented Hybrid Volleyball Premier League (AHVPL) Policy.
"""

from __future__ import annotations

import random
import time
from typing import Any, Dict, Optional

import torch
from tensordict import TensorDict

from logic.src.envs.base.base import RL4COEnvBase
from logic.src.models.policies.hgs_core.crossover import vectorized_ordered_crossover
from logic.src.models.policies.hgs_core.population import VectorizedPopulation
from logic.src.models.policies.hybrid_volleyball_premier_league import VectorizedHVPL
from logic.src.models.policies.shared.linear import vectorized_linear_split


class VectorizedAHVPL(VectorizedHVPL):
    """
    Vectorized Augmented Hybrid Volleyball Premier League (AHVPL) Policy.

    Integrates HGS genetic operators and diversity management into the
    HVPL population-based framework, optimized for GPU.
    """

    def __init__(
        self,
        env_name: str,
        n_teams: int = 10,
        max_iterations: int = 20,
        sub_rate: float = 0.2,
        time_limit: float = 60.0,
        aco_iterations: int = 1,
        alns_iterations: int = 50,
        crossover_rate: float = 0.7,
        elite_size: int = 5,
        device: str = "cuda",
        generator: Optional[torch.Generator] = None,
        rng: Optional[random.Random] = None,
        **kwargs,
    ):
        super().__init__(
            env_name=env_name,
            n_teams=n_teams,
            max_iterations=max_iterations,
            sub_rate=sub_rate,
            time_limit=time_limit,
            aco_iterations=aco_iterations,
            alns_iterations=alns_iterations,
            device=device,
            generator=generator,
            rng=rng,
            **kwargs,
        )
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size

    def __getstate__(self) -> Dict[str, Any]:
        """Return state for pickling."""
        state = self.__dict__.copy()
        # Handle torch.Generator which is not picklable
        if "generator" in state:
            gen = state["generator"]
            state["generator_state"] = gen.get_state()
            state["generator_device"] = str(gen.device)
            del state["generator"]
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restore state after pickling."""
        if "generator_state" in state:
            gen_state = state.pop("generator_state")
            gen_device = state.pop("generator_device")
            # Create a new generator on the same device
            gen = torch.Generator(device=gen_device)
            gen.set_state(gen_state)
            state["generator"] = gen
        self.__dict__.update(state)

    def forward(
        self,
        td: TensorDict,
        env: Optional[RL4COEnvBase] = None,
        strategy: str = "greedy",
        num_starts: int = 1,
        max_steps: Optional[int] = None,
        phase: str = "train",
        return_actions: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Run vectorized AHVPL.
        """
        batch_size = td.batch_size[0]
        device = td.device
        start_time = time.time()

        # 1. Setup Data
        dist_matrix, waste, capacity = self._setup_data(td)
        num_nodes = dist_matrix.shape[1]

        # 2. League Initialization (ACO Construction)
        tau = torch.ones((batch_size, num_nodes, num_nodes), device=device) * 0.1
        eta = 1.0 / (dist_matrix + 1e-6)

        # Giant tours: (batch_size, n_teams, num_nodes-1)
        population_giant = self._aco_construct(dist_matrix, tau, eta, self.n_teams)

        # Expanded data for internal solvers
        expanded_dist = dist_matrix.repeat_interleave(self.n_teams, dim=0)
        expanded_waste = waste.repeat_interleave(self.n_teams, dim=0)
        expanded_capacity = capacity.repeat_interleave(self.n_teams, dim=0)

        # Initialize Population Manager (HGS Core)
        pop_manager = VectorizedPopulation(self.n_teams, device, self.generator)

        # Initial costs (required for population management)
        # Flatten to (batch_size * n_teams, num_nodes-1)
        flat_giant = population_giant.view(batch_size * self.n_teams, -1)
        _, init_costs = vectorized_linear_split(flat_giant, expanded_dist, expanded_waste, expanded_capacity)
        pop_manager.initialize(population_giant, init_costs.view(batch_size, self.n_teams), self.elite_size)

        best_tours = population_giant[:, 0].clone()
        best_costs = torch.full((batch_size,), float("inf"), device=device)

        # 3. League Season
        for _iter in range(self.max_iterations):
            if time.time() - start_time > self.time_limit:
                break

            # 4. Learning Phase: Genetic Crossover (HGS Integration)
            # Select parents and produce offspring
            p1, p2 = pop_manager.get_parents(n_offspring=self.n_teams)

            # Crossover logic
            if torch.rand(1, device=device, generator=self.generator).item() < self.crossover_rate:
                offspring_giant = vectorized_ordered_crossover(
                    p1.view(batch_size * self.n_teams, -1),
                    p2.view(batch_size * self.n_teams, -1),
                    device=device,
                    generator=self.generator,
                ).view(batch_size, self.n_teams, -1)
            else:
                offspring_giant = p1

            # 5. Coaching Phase: ALNS Improvement
            instance_costs, coached_routes_list = self._coaching_phase(
                offspring_giant, expanded_dist, expanded_waste, expanded_capacity
            )

            # 6. Global Competition: Update Best
            improved = self._global_competition(instance_costs, coached_routes_list, best_tours, best_costs, num_nodes)

            # 7. Survivor Selection: Update Population State
            # AHVPL uses coached solutions for survivor selection
            coached_giant = self._routes_to_giant_tours(
                coached_routes_list, batch_size, self.n_teams, num_nodes - 1, offspring_giant
            )
            pop_manager.add_individuals(coached_giant, instance_costs, self.elite_size)

            # 8. Pheromone Update
            if improved.any():
                tau = self._update_pheromones(tau, best_tours, best_costs)

            # 9. VPL Substitution Phase
            if self.sub_rate > 0:
                self._ahvpl_substitution(pop_manager, dist_matrix, waste, capacity, tau, eta)

        # 10. Final packaging
        return self._format_output(td, best_tours, dist_matrix, waste, capacity, env=env)

    def _ahvpl_substitution(
        self,
        pop_manager: VectorizedPopulation,
        dist_matrix: torch.Tensor,
        waste: torch.Tensor,
        capacity: torch.Tensor,
        tau: torch.Tensor,
        eta: torch.Tensor,
    ) -> None:
        """Standard VPL substitution using ACO for worst teams."""
        batch_size = dist_matrix.shape[0]
        n_sub = int(self.n_teams * self.sub_rate)
        if n_sub <= 0:
            return

        # Identify worst indices by biased fitness
        worst_indices = pop_manager.biased_fitness.argsort(dim=1, descending=True)[:, :n_sub]

        # New constructions
        sub_tours = self._aco_construct(dist_matrix, tau, eta, n_sub)

        # Evaluate new constructions
        flat_sub = sub_tours.view(batch_size * n_sub, -1)
        expanded_dist = dist_matrix.repeat_interleave(n_sub, dim=0)
        expanded_waste = waste.repeat_interleave(n_sub, dim=0)
        expanded_capacity = capacity.repeat_interleave(n_sub, dim=0)

        _, sub_costs = vectorized_linear_split(flat_sub, expanded_dist, expanded_waste, expanded_capacity)

        # Replace worst in population manager
        for b in range(batch_size):
            for s in range(n_sub):
                idx = int(worst_indices[b, s].item())
                pop_manager.population[b, idx] = sub_tours[b, s]
                pop_manager.costs[b, idx] = sub_costs.view(batch_size, n_sub)[b, s]

        # Recompute fitness after substitution
        pop_manager.compute_biased_fitness(self.elite_size)

    def _routes_to_giant_tours(self, routes_list, batch_size, n_teams, N, backup_giant):
        """Utility to convert coached route lists back to giant tours."""
        device = backup_giant.device
        giant_tours = torch.zeros((batch_size, n_teams, N), dtype=torch.long, device=device)

        for b in range(batch_size):
            for t in range(n_teams):
                idx = b * n_teams + t
                routes = routes_list[idx]
                if isinstance(routes, torch.Tensor):
                    nodes = routes[routes != 0]
                else:
                    nodes = torch.tensor([n for n in routes if n != 0], device=device)

                # Standardize length
                if nodes.size(0) < N:
                    nodes = torch.cat([nodes, torch.zeros(N - nodes.size(0), dtype=torch.long, device=device)])
                elif nodes.size(0) > N:
                    nodes = nodes[:N]

                # If everything went wrong, fallback
                if nodes.sum() == 0:
                    giant_tours[b, t] = backup_giant[b, t].clone()
                else:
                    giant_tours[b, t] = nodes
        return giant_tours
