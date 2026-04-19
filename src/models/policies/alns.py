"""
ALNS Policy wrapper for RL4CO using vectorized implementation.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
from tensordict import TensorDict

from logic.src.constants.simulation import VEHICLE_CAPACITY
from logic.src.envs.base.base import RL4COEnvBase
from logic.src.models.common.autoregressive.policy import AutoregressivePolicy
from logic.src.models.policies.hgs import VectorizedHGS

from .adaptive_large_neighborhood_search import VectorizedALNS as VectorizedALNSEngine


class VectorizedALNS(AutoregressivePolicy):
    """
    ALNS-based Policy wrapper using vectorized GPU-accelerated implementation.
    """

    def __init__(
        self,
        env_name: str,
        time_limit: float = 5.0,
        max_iterations: int = 500,
        max_vehicles: int = 0,
        start_temp: float = 0.5,
        cooling_rate: float = 0.9995,
        device: str = "cpu",
        seed: int = 42,
        **kwargs,
    ):
        """Initialize ALNSPolicy."""
        super().__init__(env_name=env_name, device=device, seed=seed, **kwargs)
        self.time_limit = time_limit
        self.max_iterations = max_iterations
        self.max_vehicles = max_vehicles
        self.start_temp = start_temp
        self.cooling_rate = cooling_rate

    def forward(
        self,
        td: TensorDict,
        env: Optional[RL4COEnvBase] = None,
        strategy: str = "greedy",  # Ignored for ALNS
        num_starts: int = 1,
        max_steps: Optional[int] = None,
        phase: str = "train",
        return_actions: bool = True,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Solve instances in the batch using vectorized ALNS.
        """
        batch_size = td.batch_size[0]
        device = td.device

        # Extract data
        customers = td["locs"]  # (batch, num_nodes, 2)
        depot = td["depot"].unsqueeze(1)  # (batch, 1, 2)
        locs = torch.cat([depot, customers], dim=1)  # (batch, num_nodes + 1, 2)

        device = locs.device
        num_nodes = locs.shape[1]

        # Compute distance matrix if needed
        if locs.dim() == 3 and locs.shape[-1] == 2:
            # Compute Euclidean distance matrix
            diff = locs.unsqueeze(2) - locs.unsqueeze(1)
            dist_matrix = torch.sqrt((diff**2).sum(dim=-1))
        else:
            dist_matrix = locs

        # Extract waste
        waste_at_nodes = td.get("waste", torch.zeros(batch_size, num_nodes - 1, device=device))
        # Prepend 0 for depot
        waste = torch.cat([torch.zeros(batch_size, 1, device=device), waste_at_nodes], dim=1)

        # Extract capacity
        capacity = td.get("capacity", torch.ones(batch_size, device=device) * VEHICLE_CAPACITY)
        if capacity.dim() == 0:
            capacity = capacity.expand(batch_size)

        # Create initial solutions (random permutations)
        initial_solutions = torch.stack(
            [torch.randperm(num_nodes - 1, device=device, generator=self.generator) + 1 for _ in range(batch_size)]
        )

        solver = VectorizedALNSEngine(
            dist_matrix=dist_matrix,
            wastes=waste,
            vehicle_capacity=capacity,
            time_limit=self.time_limit,
            device=str(device),
            generator=self.generator,
        )

        routes_list, costs = solver.solve(
            initial_solutions=initial_solutions,
            n_iterations=self.max_iterations,
            time_limit=self.time_limit,
            max_vehicles=kwargs.get("max_vehicles", self.max_vehicles),
            start_temp=self.start_temp,
            cooling_rate=self.cooling_rate,
        )

        # Convert routes to actions (padded tensor)
        all_actions = []
        for b in range(batch_size):
            route_nodes_raw = routes_list[b]
            # Ensure it starts and ends with 0 if not already
            if isinstance(route_nodes_raw, torch.Tensor):
                route_tensor = route_nodes_raw
            else:
                route_tensor = torch.tensor(route_nodes_raw, device=device)

            # vectorized_linear_split usually returns routes with 0s.
            # If not, we add them.
            actions = route_tensor if route_tensor.size(0) > 0 else torch.tensor([0, 0], device=device)
            all_actions.append(actions)

        # Pad actions
        max_len = max(len(a) for a in all_actions)
        padded_actions = torch.stack(
            [torch.cat([a, torch.zeros(max_len - len(a), device=device, dtype=torch.long)]) for a in all_actions]
        )

        # Compute reward using the same cost function as the model
        reward = VectorizedHGS._compute_reward(td, env, padded_actions)

        return {
            "reward": reward,
            "actions": padded_actions,
            "log_likelihood": torch.zeros(batch_size, device=device),
            "cost": costs,
        }
