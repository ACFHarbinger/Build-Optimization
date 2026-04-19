"""
HGS Policy wrapper for RL4CO using vectorized implementation.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from tensordict import TensorDict

from logic.src.constants.simulation import VEHICLE_CAPACITY
from logic.src.envs.base.base import RL4COEnvBase
from logic.src.models.common.autoregressive.policy import AutoregressivePolicy
from logic.src.models.policies.hybrid_genetic_search import VectorizedHGS as VectorizedHGSEngine


class VectorizedHGS(AutoregressivePolicy):
    """
    HGS-based Policy wrapper using vectorized GPU-accelerated implementation.
    """

    def __init__(
        self,
        env_name: str,
        time_limit: float = 5.0,
        population_size: int = 50,
        n_generations: int = 50,
        elite_size: int = 10,
        max_vehicles: int = 0,
        crossover_rate: float = 0.7,
        max_iterations: int = 50,
        seed: int = 42,
        device: str = "cpu",
        **kwargs,
    ):
        """Initialize HGSPolicy with vectorized solver."""
        super().__init__(env_name=env_name, seed=seed, device=device, **kwargs)
        self.time_limit = time_limit
        self.population_size = population_size
        self.n_generations = n_generations
        self.elite_size = elite_size
        self.max_vehicles = max_vehicles
        self.crossover_rate = crossover_rate
        self.max_iterations = max_iterations

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
        Solve instances in the batch using vectorized HGS.
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

        # 3. Solver Execution
        solver = VectorizedHGSEngine(
            dist_matrix=dist_matrix,
            wastes=waste,
            vehicle_capacity=capacity,
            max_iterations=self.max_iterations,
            time_limit=self.time_limit,
            device=str(device),
            generator=self.generator,
            rng=self.rng,
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
        # Vectorized linear split output reconstruction (padded)
        all_actions = []
        for b in range(batch_size):
            routes = best_routes_list[b]
            # Convert to flat action sequence with depot 0
            a = routes if isinstance(routes, torch.Tensor) else torch.tensor(routes, device=device, dtype=torch.long)
            all_actions.append(a)

        max_len = max([len(a) for a in all_actions] + [2])
        padded_actions = torch.stack(
            [torch.cat([a, torch.zeros(max_len - len(a), device=device, dtype=torch.long)]) for a in all_actions]
        )

        # Compute reward using the same cost function as the model
        reward = self._compute_reward(td, env, padded_actions)

        return {
            "actions": padded_actions,
            "reward": reward,
            "cost": costs.to(device),
            "log_likelihood": torch.zeros(batch_size, device=device),
        }

    @staticmethod
    def _compute_reward(td: TensorDict, env: Optional[RL4COEnvBase], actions: torch.Tensor) -> torch.Tensor:
        """Replay actions through the environment to compute the exact same
        reward as the neural model (via ``env.get_reward``)."""
        if env is None:
            return -torch.ones(td.batch_size[0], device=actions.device)

        with torch.no_grad():
            curr_td = env.reset(td)
            for i in range(actions.size(1)):
                curr_td["action"] = actions[:, i]
                curr_td = env.step(curr_td)["next"]
            return env.get_reward(curr_td, actions)
