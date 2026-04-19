"""
Iterated Local Search (ILS) Policy.

A metaheuristic expert policy that combines local search with perturbation
to escape local minima. Useful for imitation learning and as a strong baseline.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import torch
from tensordict import TensorDict

from logic.src.envs.base.base import RL4COEnvBase
from logic.src.models.common.improvement.policy import ImprovementPolicy
from logic.src.tracking.viz_mixin import PolicyVizMixin

from .local_search import (
    vectorized_relocate,
    vectorized_swap,
    vectorized_swap_star,
    vectorized_three_opt,
    vectorized_two_opt,
    vectorized_two_opt_star,
)
from .shared.linear import vectorized_linear_split


class IteratedLocalSearchPolicy(ImprovementPolicy, PolicyVizMixin):
    """
    Iterated Local Search (ILS) expert policy.

    Combines local search with perturbation to escape local minima.
    The algorithm:
    1. Apply local search until local optimum
    2. Perturb the solution (double-bridge or segment shuffle)
    3. Apply local search again
    4. Accept new solution if improving
    5. Repeat for n_restarts
    """

    def __init__(
        self,
        env_name: str,
        ls_operator: Union[str, dict[str, float]] = "two_opt",
        perturbation_type: Union[str, dict[str, float]] = "double_bridge",
        n_restarts: int = 5,
        ls_iterations: int = 50,
        perturbation_strength: float = 0.2,
        seed: int = 42,
        device: str = "cpu",
        **kwargs,
    ):
        """
        Initialize IteratedLocalSearchPolicy.

        Args:
            env_name: Name of the environment.
            ls_operator: Local search operator or dict of {name: prob}.
            perturbation_type: Perturbation method or dict of {mode: prob}.
            n_restarts: Number of ILS restarts (perturbation cycles).
            ls_iterations: Iterations for local search within each phase.
            perturbation_strength: Fraction of tour to perturb (for shuffle/swap).
            **kwargs: Additional arguments for AutoregressivePolicy.
        """
        super().__init__(env_name=env_name, seed=seed, device=device, **kwargs)
        self.ls_operator = ls_operator
        self.perturbation_type = perturbation_type
        self.n_restarts = n_restarts
        self.ls_iterations = ls_iterations
        self.perturbation_strength = perturbation_strength

        # Default probabilities
        self.default_op_probs = {
            "two_opt": 0.25,
            "swap": 0.15,
            "relocate": 0.15,
            "two_opt_star": 0.2,
            "swap_star": 0.15,
            "three_opt": 0.1,
        }
        self.default_perturb_probs = {
            "double_bridge": 0.5,
            "shuffle": 0.3,
            "random_swap": 0.2,
        }

        # Mapping operator names to functions
        self.op_map = {
            "two_opt": vectorized_two_opt,
            "swap": vectorized_swap,
            "relocate": vectorized_relocate,
            "three_opt": vectorized_three_opt,
            "two_opt_star": vectorized_two_opt_star,
            "swap_star": vectorized_swap_star,
            "2opt": vectorized_two_opt,
            "3opt": vectorized_three_opt,
        }

    def _perturb(self, tours: torch.Tensor, device: torch.device, mode: str) -> torch.Tensor:
        """
        Apply perturbation to a batch of tours.

        Args:
            tours: Batch of tours [B, N].
            device: Torch device.
            mode: Perturbation mode.

        Returns:
            Perturbed tours [B, N].
        """
        B, N = tours.shape
        result = tours.clone()

        for b in range(B):
            t_list: List[int] = tours[b].tolist()
            # Remove padding (zeros at end)
            actual_n = int((tours[b] != 0).sum().item())
            # If the tour has a trailing zero that isn't padding (depot at end)
            if 0 < actual_n < N and t_list[actual_n - 1] == 0:
                pass  # already counted
            elif actual_n == N:
                pass

            n = len(t_list)

            if mode == "double_bridge":
                # Double-bridge move: break tour into 4 segments and reconnect
                if n < 8:
                    continue
                positions = sorted(self.rng.sample(range(1, n - 1), 3))
                p1, p2, p3 = positions
                new_list = t_list[:p1] + t_list[p2:p3] + t_list[p1:p2] + t_list[p3:]
                result[b, : len(new_list)] = torch.tensor(new_list, device=device)

            elif mode == "shuffle":
                seg_len = max(2, int(n * self.perturbation_strength))
                if n - seg_len - 1 <= 1:
                    continue
                start = self.rng.randint(1, n - seg_len - 1)
                segment = t_list[start : start + seg_len]
                self.rng.shuffle(segment)
                new_list = t_list[:start] + segment + t_list[start + seg_len :]
                result[b, : len(new_list)] = torch.tensor(new_list, device=device)

            else:  # random_swap
                n_swaps = max(1, int(n * self.perturbation_strength))
                new_list = t_list[:]
                for _ in range(n_swaps):
                    if n < 3:
                        break
                    i, j = self.rng.sample(range(1, n - 1), 2)
                    new_list[i], new_list[j] = new_list[j], new_list[i]
                result[b, : len(new_list)] = torch.tensor(new_list, device=device)

        return result

    def _compute_costs(self, tours: torch.Tensor, dist_matrix: torch.Tensor) -> torch.Tensor:
        """Compute tour costs for a batch."""
        B = tours.shape[0]
        device = tours.device
        batch_ids = torch.arange(B, device=device).view(B, 1)

        from_n = tours[:, :-1]
        to_n = tours[:, 1:]

        dists = dist_matrix[batch_ids, from_n, to_n] if dist_matrix.dim() == 3 else dist_matrix[from_n, to_n]

        return dists.sum(dim=1)

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
        Refine solutions using ILS.
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
                [
                    torch.randperm(num_nodes - 1, device=self.device, generator=self.generator) + 1
                    for _ in range(batch_size)
                ]
            )

        # Convert to routed format initially
        routes_list, _ = vectorized_linear_split(solutions, dist_matrix, waste, capacity)

        # Create a padded route tensor (B, max_len)
        max_l = max(len(r) for r in routes_list)
        current_routes = torch.zeros((batch_size, max_l), dtype=torch.long, device=device)
        for b in range(batch_size):
            r = routes_list[b]
            current_routes[b, : len(r)] = torch.tensor(r, device=device)

        # 3. Prepare sampling distributions
        op_probs_dict = None
        ops_sorted: list[str] = []
        op_weights: list[float] = []

        if isinstance(self.ls_operator, str) and self.ls_operator == "random":
            op_probs_dict = self.default_op_probs
        elif isinstance(self.ls_operator, dict):
            op_probs_dict = self.ls_operator

        if op_probs_dict:
            ops_sorted = sorted(op_probs_dict.keys())
            op_weights = [op_probs_dict.get(op, 0.0) for op in ops_sorted]

        p_probs_dict = None
        p_modes_sorted: list[str] = []
        p_weights: list[float] = []

        if isinstance(self.perturbation_type, str) and self.perturbation_type == "random":
            p_probs_dict = self.default_perturb_probs
        elif isinstance(self.perturbation_type, dict):
            p_probs_dict = self.perturbation_type

        if p_probs_dict:
            p_modes_sorted = sorted(p_probs_dict.keys())
            p_weights = [p_probs_dict.get(m, 0.0) for m in p_modes_sorted]

        # 4. Initial local search
        if op_probs_dict:
            initial_op = self.rng.choices(ops_sorted, weights=op_weights)[0]
            ls_func = self.op_map.get(initial_op, vectorized_two_opt)
        else:
            op_name = self.ls_operator if isinstance(self.ls_operator, str) else "two_opt"
            ls_func = self.op_map.get(op_name, vectorized_two_opt)

        current_routes = ls_func(
            current_routes, dist_matrix, max_iterations=self.ls_iterations, generator=self.generator
        )
        best_routes = current_routes.clone()
        best_costs = self._compute_costs(best_routes, dist_matrix)

        # 5. ILS loop
        for _restart_idx in range(self.n_restarts):
            # Sample perturbation mode
            if p_probs_dict:
                p_mode = self.rng.choices(p_modes_sorted, weights=p_weights)[0]
            else:
                p_mode = self.perturbation_type if isinstance(self.perturbation_type, str) else "double_bridge"

            # Perturb
            perturbed = self._perturb(current_routes, device, p_mode)

            # Sample LS operator for this restart
            if op_probs_dict:
                iter_op = self.rng.choices(ops_sorted, weights=op_weights)[0]
                iter_ls_func = self.op_map.get(iter_op, vectorized_two_opt)
            else:
                iter_ls_func = ls_func if ls_func is not None else vectorized_two_opt

            # Local search on perturbed solution
            refined = iter_ls_func(perturbed, dist_matrix, max_iterations=self.ls_iterations, generator=self.generator)
            refined_costs = self._compute_costs(refined, dist_matrix)

            # Accept if improving
            improved = refined_costs < best_costs - 1e-6
            best_routes[improved] = refined[improved]
            best_costs[improved] = refined_costs[improved]

            self._viz_record(
                restart=_restart_idx,
                perturb_mode=p_mode,
                n_improved=int(improved.sum().item()),
                best_cost=float(best_costs.min().item()),
                candidate_cost=float(refined_costs.min().item()),
            )

            # Always move to refined for exploration
            current_routes = refined

        # 6. Final Evaluation and Output
        costs = best_costs

        # RL4CO return format
        R = getattr(env, "waste_weight", 1.0)
        C = getattr(env, "cost_weight", 1.0)

        all_rewards = []
        for b in range(batch_size):
            collected_nodes = set(best_routes[b].tolist()) - {0}
            profit = sum(waste[b, node].item() * R for node in collected_nodes if node < num_nodes)
            cost = costs[b].item() * C
            all_rewards.append(torch.tensor(profit - cost, device=device))

        return {
            "reward": torch.stack(all_rewards),
            "actions": best_routes,
            "log_likelihood": torch.zeros(batch_size, device=device),
            "cost": costs,
        }
