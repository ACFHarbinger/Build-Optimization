"""
GLOP Policy.

Global-Local Optimization Policy combining NAR partitioning with local solvers.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Literal, Optional

import torch
from tensordict import TensorDict

from logic.src.envs.base.base import RL4COEnvBase
from logic.src.models.common.non_autoregressive.policy import NonAutoregressivePolicy
from logic.src.models.subnets.modules.glop_factory import get_adapter

# Type for subproblem solvers
SubProblemSolverType = Callable[[torch.Tensor], torch.Tensor]


class GLOPPolicy(NonAutoregressivePolicy):
    """
    GLOP: Global-Local Optimization Policy.

    Two-stage hierarchical approach:
    1. Global: Use NAR encoder to partition nodes into groups
    2. Local: Solve each partition with a local solver (insertion, neural TSP)

    Reference:
        Ye et al. "GLOP: Learning Global Partition and Local Construction
        for Solving Large-scale Routing Problems in Real-time" (2023)
    """

    def __init__(
        self,
        env_name: str = "cvrp",
        n_samples: int = 10,
        temperature: float = 1.0,
        embed_dim: int = 64,
        subprob_solver: SubProblemSolverType | str = "greedy",
        subprob_batch_size: int = 2000,
        **encoder_kwargs,
    ) -> None:
        """
        Initialize GLOP policy.

        Args:
            env_name: Environment name.
            n_samples: Number of samples for multistart.
            temperature: Temperature for sampling.
            embed_dim: Embedding dimension.
            subprob_solver: Solver for local subproblems.
            subprob_batch_size: Batch size for subproblem solving.
            **encoder_kwargs: Additional encoder arguments.
        """
        super().__init__(
            env_name=env_name,
            embed_dim=embed_dim,
            temperature=temperature,
            train_strategy="multistart_sampling",
            val_strategy="multistart_greedy",
            test_strategy="multistart_greedy",
            **encoder_kwargs,
        )

        self.n_samples = n_samples
        self.subprob_solver = subprob_solver
        self.subprob_batch_size = subprob_batch_size

        # Get appropriate adapter for environment
        self.adapter_class = get_adapter(env_name)

    def forward(  # type: ignore[override]
        self,
        td: TensorDict,
        env: Optional[RL4COEnvBase] = None,
        phase: Literal["train", "val", "test"] = "test",
        calc_reward: bool = True,
        return_actions: bool = False,
        return_entropy: bool = False,
        strategy: Optional[str] = None,
        **decoding_kwargs,
    ) -> Dict[str, Any]:
        """
        Forward pass with global partitioning + local solving.

        Args:
            td: TensorDict with problem data.
            env: Environment instance.
            phase: Current phase.
            calc_reward: Whether to calculate reward.
            return_actions: Whether to return actions.
            return_entropy: Whether to return entropy.

        Returns:
            Dictionary with reward, log_likelihood, and optionally actions.
        """
        # Stage 1: Global partitioning via NAR policy
        par_out = super().forward(
            td=td,
            env=env,  # type: ignore[arg-type]
            phase=phase,
            calc_reward=False,
            return_actions=True,  # Need partition actions
            return_entropy=return_entropy,
            num_starts=self.n_samples,
            strategy=strategy,  # type: ignore[arg-type]
            **decoding_kwargs,
        )

        partition_actions = par_out["actions"]

        # Stage 2: Local solving
        local_out = self._local_policy(td, partition_actions)
        final_actions = local_out["actions"]

        # Build output
        out = par_out
        if return_actions:
            out["actions"] = final_actions
        else:
            out.pop("actions", None)

        # Calculate reward on final solution
        if calc_reward and env is not None:
            from logic.src.utils.decoding import batchify

            td_repeated = batchify(td, self.n_samples)
            reward = env.get_reward(td_repeated, final_actions)
            out["reward"] = reward.detach()

        return out

    @torch.no_grad()
    def _local_policy(
        self,
        td: TensorDict,
        partition_actions: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Apply local solver to each partition.

        Args:
            td: Problem TensorDict.
            partition_actions: Partition assignments.

        Returns:
            Dictionary with refined actions.
        """
        # Reshape for n_samples handling
        # (n_samples * batch) -> (batch * n_samples)
        from logic.src.utils.decoding import unbatchify

        partition_actions_reshaped = (
            unbatchify(partition_actions, self.n_samples)
            .transpose(0, 1)
            .contiguous()
            .view(-1, partition_actions.size(-1))
        )

        # Create adapter
        adapter = self.adapter_class(
            td,
            partition_actions_reshaped,
            subprob_batch_size=self.subprob_batch_size,
        )

        # Solve subproblems
        solver = self._get_solver()
        for mapping in adapter.get_batched_subprobs():
            subprob_actions = solver(mapping.subprob_coordinates)
            adapter.update_actions(mapping, subprob_actions)

        # Get final actions and reshape back
        actions = adapter.get_actions().to(td.device)
        batch_size = td.batch_size[0] if td.batch_size else partition_actions.size(0) // self.n_samples
        actions = actions.view(batch_size, self.n_samples, -1).transpose(0, 1).contiguous()
        actions = actions.view(-1, actions.size(-1))

        return {"actions": actions}

    def _get_solver(self) -> SubProblemSolverType:
        """Get the subproblem solver function."""
        if isinstance(self.subprob_solver, str):
            if self.subprob_solver == "greedy":
                return self._greedy_tsp_solver
            elif self.subprob_solver == "nearest":
                return self._nearest_neighbor_solver
            else:
                raise ValueError(f"Unknown solver: {self.subprob_solver}")
        return self.subprob_solver

    @staticmethod
    def _greedy_tsp_solver(coords: torch.Tensor) -> torch.Tensor:
        """
        Simple greedy TSP solver.

        Args:
            coords: Coordinates (num_nodes, 2).

        Returns:
            Tour as node indices.
        """
        n = coords.size(0)
        if n <= 1:
            return torch.arange(n)

        visited = torch.zeros(n, dtype=torch.bool)
        tour = [0]
        visited[0] = True

        for _ in range(n - 1):
            current = tour[-1]
            # Find nearest unvisited
            dists = torch.cdist(coords[current : current + 1], coords).squeeze(0)
            dists[visited] = float("inf")
            nearest = dists.argmin().item()
            tour.append(nearest)  # type: ignore[arg-type]
            visited[nearest] = True  # type: ignore[index]

        return torch.tensor(tour)

    @staticmethod
    def _nearest_neighbor_solver(coords: torch.Tensor) -> torch.Tensor:
        """Nearest neighbor heuristic (alias for greedy)."""
        return GLOPPolicy._greedy_tsp_solver(coords)
