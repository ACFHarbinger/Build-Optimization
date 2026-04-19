"""
Vectorized HGS-ALNS Hybrid Policy for RL.
"""

from __future__ import annotations

import random
from typing import Any, Dict, Optional, Tuple

import torch
from tensordict import TensorDict

from logic.src.constants.simulation import VEHICLE_CAPACITY
from logic.src.envs.base.base import RL4COEnvBase
from logic.src.models.policies.adaptive_large_neighborhood_search import VectorizedALNS
from logic.src.models.policies.hgs import VectorizedHGS as VectorizedHGSPolicy
from logic.src.models.policies.hybrid_genetic_search import VectorizedHGS as VectorizedHGSEngine


class VectorizedHGSALNSEngine(VectorizedHGSEngine):
    """
    HGS Engine that uses ALNS for the Education phase.
    """

    def __init__(
        self,
        dist_matrix: torch.Tensor,
        wastes: torch.Tensor,
        vehicle_capacity: Any,
        time_limit: float = 1.0,
        device: str = "cpu",
        rng: Optional[random.Random] = None,
        generator: Optional[torch.Generator] = None,
        alns_education_iterations: int = 50,
        alns_start_temp: float = 0.5,
        alns_cooling_rate: float = 0.9995,
    ):
        super().__init__(
            dist_matrix=dist_matrix,
            wastes=wastes,
            vehicle_capacity=vehicle_capacity,
            time_limit=time_limit,
            device=device,
            generator=generator,
            rng=rng,
        )
        self.alns_education_iterations = alns_education_iterations
        self.alns_start_temp = alns_start_temp
        self.alns_cooling_rate = alns_cooling_rate
        self.alns_engine = VectorizedALNS(
            dist_matrix=dist_matrix,
            wastes=wastes,
            vehicle_capacity=vehicle_capacity,
            device=device,
            time_limit=self.time_limit,
        )

    def educate(
        self, routes_list: list[list[int]], split_costs: torch.Tensor, max_vehicles: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Education Phase using Vectorized ALNS.
        """
        B = len(routes_list)
        N = self.dist_matrix.shape[1] - 1  # Total customers

        # 1. Convert initial routes (from linear split) to giant tour format for ALNS
        # ALNS expects giant tours of size N.
        initial_solutions = torch.zeros((B, N), dtype=torch.long, device=self.device)
        for b in range(B):
            r = routes_list[b]
            nodes = torch.tensor([n for n in r if n != 0], device=self.device, dtype=torch.long)
            if nodes.size(0) > N:
                nodes = nodes[:N]
            elif nodes.size(0) < N:
                # Pad if necessary (unlikely given linear split on full tour)
                nodes = torch.cat([nodes, torch.zeros(N - nodes.size(0), device=self.device, dtype=torch.long)])
            initial_solutions[b] = nodes

        # 2. Run Vectorized ALNS
        # ALNS.solve returns (best_routes_list, costs)
        # Note: VectorizedALNS.solve is primarily for independent instances, but here
        # we treat the genetic offspring as instances.
        improved_routes_list, improved_costs = self.alns_engine.solve(
            initial_solutions=initial_solutions,
            n_iterations=self.alns_education_iterations,
            max_vehicles=max_vehicles,
            start_temp=self.alns_start_temp,
            cooling_rate=self.alns_cooling_rate,
        )

        # 3. Format as padded routes tensor (required by HGS loop)
        max_l = max(len(r) for r in improved_routes_list)
        improved_routes_tensor = torch.zeros((B, max_l), dtype=torch.long, device=self.device)
        for b in range(B):
            r = improved_routes_list[b]
            improved_routes_tensor[b, : len(r)] = torch.tensor(r, device=self.device)

        return improved_routes_tensor, improved_costs


class VectorizedHGSALNS(VectorizedHGSPolicy):
    """
    Vectorized HGS-ALNS Policy wrapper for RL4CO.
    """

    def __init__(
        self,
        env_name: str,
        time_limit: float = 5.0,
        population_size: int = 20,
        n_generations: int = 10,
        elite_size: int = 5,
        crossover_rate: float = 0.5,
        max_vehicles: int = 0,
        alns_education_iterations: int = 50,
        alns_start_temp: float = 0.5,
        alns_cooling_rate: float = 0.9995,
        seed: int = 42,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(
            env_name=env_name,
            time_limit=time_limit,
            population_size=population_size,
            n_generations=n_generations,
            elite_size=elite_size,
            max_vehicles=max_vehicles,
            crossover_rate=crossover_rate,
            seed=seed,
            device=device,
            **kwargs,
        )
        self.alns_education_iterations = alns_education_iterations
        self.alns_start_temp = alns_start_temp
        self.alns_cooling_rate = alns_cooling_rate

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
        Solve instances in the batch using vectorized HGS-ALNS.
        """
        batch_size = td.batch_size[0]
        device = td.device if td.device is not None else td["locs"].device

        # 1. Setup Data
        locs = td["locs"]
        num_nodes = locs.shape[1]
        if locs.dim() == 3 and locs.shape[-1] == 2:
            diff = locs.unsqueeze(2) - locs.unsqueeze(1)
            dist_matrix = torch.sqrt((diff**2).sum(dim=-1))
        else:
            dist_matrix = locs

        waste_at_nodes = td.get("waste", torch.zeros(batch_size, num_nodes - 1, device=device))
        waste = torch.cat([torch.zeros(batch_size, 1, device=device), waste_at_nodes], dim=1)
        capacity = td.get("capacity", torch.ones(batch_size, device=device) * VEHICLE_CAPACITY)
        if capacity.dim() == 0:
            capacity = capacity.expand(batch_size)

        # 2. Initialization: Random permutations
        initial_solutions = torch.stack(
            [torch.randperm(num_nodes - 1, device=device, generator=self.generator) + 1 for _ in range(batch_size)]
        )

        # 3. Solver Execution using HGS-ALNS Engine
        solver = VectorizedHGSALNSEngine(
            dist_matrix=dist_matrix,
            wastes=waste,
            vehicle_capacity=capacity,
            time_limit=self.time_limit,
            device=str(device),
            rng=self.rng,
            generator=self.generator,
            alns_education_iterations=self.alns_education_iterations,
            alns_start_temp=self.alns_start_temp,
            alns_cooling_rate=self.alns_cooling_rate,
        )

        best_routes_list, costs = solver.solve(
            initial_solutions=initial_solutions,
            n_generations=self.n_generations,
            population_size=self.population_size,
            elite_size=self.elite_size,
            max_vehicles=self.max_vehicles,
            time_limit=self.time_limit,
            crossover_rate=self.crossover_rate,
        )

        # 4. Format Actions
        all_actions = []
        for b in range(batch_size):
            routes = best_routes_list[b]
            a = routes if isinstance(routes, torch.Tensor) else torch.tensor(routes, device=device, dtype=torch.long)
            all_actions.append(a)

        max_len = max([len(a) for a in all_actions] + [2])
        padded_actions = torch.stack(
            [torch.cat([a, torch.zeros(max_len - len(a), device=device, dtype=torch.long)]) for a in all_actions]
        )

        # Compute reward using the same cost function as the model
        reward = VectorizedHGSPolicy._compute_reward(td, env, padded_actions)

        return {
            "actions": padded_actions,
            "reward": reward,
            "cost": costs.to(device),
            "log_likelihood": torch.zeros(batch_size, device=device),
        }
