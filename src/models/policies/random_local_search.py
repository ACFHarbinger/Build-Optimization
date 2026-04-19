"""
Random Local Search Policy expert.
Performs iterative local search moves sampled from a set of operators based on provided probabilities.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from tensordict import TensorDict

from logic.src.envs.base.base import RL4COEnvBase
from logic.src.models.common.improvement.policy import ImprovementPolicy
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


class RandomLocalSearchPolicy(ImprovementPolicy, PolicyVizMixin):
    """
    Random Local Search expert policy.

    Iteratively applies local search operators sampled from a probability distribution.
    Useful for imitation learning and adaptive refinement.
    """

    def __init__(
        self,
        env_name: str,
        n_iterations: int = 100,
        op_probs: dict[str, float] | None = None,
        seed: int = 42,
        device: str = "cpu",
        **kwargs,
    ):
        """
        Initialize RandomLocalSearchPolicy.

        Args:
            env_name: Name of the environment.
            n_iterations: Number of search iterations (operator applications).
            op_probs: Dictionary mapping operator names to selection probabilities.
                     Normalized internally if they don't sum to 1.
                     Supported keys: 'two_opt', 'swap', 'relocate', 'two_opt_star', 'swap_star', 'three_opt'.
            seed: Random seed for reproducibility.
            device: Device to use for computation.
            **kwargs: Additional arguments for ConstructivePolicy.
        """
        super().__init__(env_name=env_name, seed=seed, device=device, **kwargs)
        self.n_iterations = n_iterations

        # Default probabilities (weighted towards more powerful or faster operators)
        default_probs = {
            "two_opt": 0.25,
            "swap": 0.15,
            "relocate": 0.15,
            "two_opt_star": 0.2,
            "swap_star": 0.15,
            "three_opt": 0.1,
        }
        self.op_probs_dict = op_probs if op_probs is not None else default_probs

        # Normalize and prepare for sampling
        ops = sorted(self.op_probs_dict.keys())
        probs = torch.tensor([self.op_probs_dict[op] for op in ops], dtype=torch.float32)
        self.probs = probs / probs.sum()
        self.ops = ops

        # Mapping operator names to functions
        self.op_map = {
            "two_opt": vectorized_two_opt,
            "swap": vectorized_swap,
            "relocate": vectorized_relocate,
            "two_opt_star": vectorized_two_opt_star,
            "swap_star": vectorized_swap_star,
            "three_opt": vectorized_three_opt,
        }

    def forward(
        self,
        td: TensorDict,
        env: Optional[RL4COEnvBase] = None,
        strategy: str = "greedy",  # Ignored
        num_starts: int = 1,
        max_steps: Optional[int] = None,
        phase: str = "train",
        return_actions: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Refine solutions using stochastic local search.
        """
        batch_size = td.batch_size[0]
        device = td.device

        # 1. Extract environment data
        locs = td["locs"]
        device = td.device if td.device is not None else locs.device
        num_nodes = locs.shape[1]

        if locs.dim() == 3 and locs.shape[-1] == 2:
            diff = locs.unsqueeze(2) - locs.unsqueeze(1)
            dist_matrix = torch.sqrt((diff**2).sum(dim=-1))
        else:
            dist_matrix = locs

        waste = td.get("waste", torch.zeros(batch_size, num_nodes, device=device))
        capacity = td.get("capacity", torch.ones(batch_size, device=device))

        # 2. Prepare initial solutions (giant tours)
        if kwargs.get("initial_solution") is not None:
            solutions = kwargs["initial_solution"].clone().to(device)
        else:
            solutions = torch.stack(
                [torch.randperm(num_nodes - 1, device=device, generator=self.generator) + 1 for _ in range(batch_size)]
            )

        # Convert to routed format initially
        routes_list, _ = vectorized_linear_split(solutions, dist_matrix, waste, capacity)

        # Create a padded route tensor (B, max_len)
        max_l = max(len(r) for r in routes_list)
        current_routes = torch.zeros((batch_size, max_l), dtype=torch.long, device=device)
        for b in range(batch_size):
            r = routes_list[b]
            current_routes[b, : len(r)] = torch.tensor(r, device=device)

        # 3. Iterative Stochastic Local Search
        # Pre-sample operators for all iterations
        self.probs = self.probs.to(device)

        op_indices = torch.multinomial(
            self.probs, self.n_iterations, replacement=True, generator=self.generator
        ).tolist()

        for _rls_iter, op_idx in enumerate(op_indices):
            op_name = self.ops[op_idx]
            op_func = self.op_map[op_name]

            # Apply operator
            # We set max_iterations=1 to ensure we respect our n_iterations and stochasticity
            current_routes = op_func(current_routes, dist_matrix, max_iterations=1, generator=self.generator)

            self._viz_record(iteration=_rls_iter, op_name=op_name)

        # 4. Final Evaluation and Output
        from_n = current_routes[:, :-1]
        to_n = current_routes[:, 1:]

        batch_ids = torch.arange(batch_size, device=device).view(batch_size, 1)
        # Correctly handle 3D dist_matrix
        dists = dist_matrix[batch_ids, from_n, to_n] if dist_matrix.dim() == 3 else dist_matrix[from_n, to_n]

        costs = dists.sum(dim=1)

        # RL4CO return format
        # We return the routed result as actions.

        # Compute rewards (profit - cost)
        R = getattr(env, "waste_weight", 1.0)
        C = getattr(env, "cost_weight", 1.0)

        all_rewards = []
        for b in range(batch_size):
            collected_nodes = set(current_routes[b].tolist()) - {0}
            profit = sum(waste[b, node].item() * R for node in collected_nodes if node < num_nodes)
            cost = costs[b].item() * C
            all_rewards.append(torch.tensor(profit - cost, device=device))

        return {
            "reward": torch.stack(all_rewards),
            "actions": current_routes,
            "log_likelihood": torch.zeros(batch_size, device=device),
            "cost": costs,
        }

    def __getstate__(self):
        """Prepare state for pickling (handle non-picklable Generator)."""
        state = self.__dict__.copy()
        state["generator_state"] = self.generator.get_state()
        state["generator_device"] = str(self.generator.device)
        del state["generator"]
        return state

    def __setstate__(self, state):
        """Restore state after unpickling."""
        gen_state = state.pop("generator_state")
        gen_device = state.pop("generator_device")
        self.__dict__.update(state)
        self.generator = torch.Generator(device=gen_device)
        self.generator.set_state(gen_state)
