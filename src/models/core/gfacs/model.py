"""
GFACS Model.

GFlowNet Ant Colony System with Trajectory Balance loss.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, cast

import numpy as np
import scipy.special
import torch
from tensordict import TensorDict
from torch import nn

from logic.src.envs.base.base import RL4COEnvBase

from .policy import GFACSPolicy


class GFACS(nn.Module):
    """
    Implements GFACS (GFlowNet Ant Colony System).

    Reference: https://arxiv.org/abs/2403.07041

    GFACS uses Trajectory Balance (TB) loss for training, which requires
    computing forward and backward probabilities for each trajectory.

    Args:
        env: Environment to use for the algorithm.
        policy: Policy to use for the algorithm. If None, creates default GFACSPolicy.
        baseline: Baseline type (currently unused, kept for compatibility).
        train_with_local_search: Whether to train with local search.
        policy_kwargs: Keyword arguments for policy.
        baseline_kwargs: Keyword arguments for baseline (currently unused).
        alpha_min: Minimum value for alpha coefficient.
        alpha_max: Maximum value for alpha coefficient.
        alpha_flat_epochs: Number of epochs to keep alpha constant.
        beta_min: Minimum value for beta coefficient.
        beta_max: Maximum value for beta coefficient.
        beta_flat_epochs: Number of epochs to keep beta constant.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        env: RL4COEnvBase,
        policy: Optional[GFACSPolicy] = None,
        baseline: str = "no",
        train_with_local_search: bool = True,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        baseline_kwargs: Optional[Dict[str, Any]] = None,
        alpha_min: float = 0.5,
        alpha_max: float = 1.0,
        alpha_flat_epochs: int = 5,
        beta_min: float = 1.0,
        beta_max: float = 1.0,
        beta_flat_epochs: int = 5,
        **kwargs,
    ) -> None:
        """Initialize Class.

        Args:
            env (RL4COEnvBase): Description of env.
            policy (Optional[GFACSPolicy]): Description of policy.
            baseline (str): Description of baseline.
            train_with_local_search (bool): Description of train_with_local_search.
            policy_kwargs (Optional[Dict[str, Any]]): Description of policy_kwargs.
            baseline_kwargs (Optional[Dict[str, Any]]): Description of baseline_kwargs.
            alpha_min (float): Description of alpha_min.
            alpha_max (float): Description of alpha_max.
            alpha_flat_epochs (int): Description of alpha_flat_epochs.
            beta_min (float): Description of beta_min.
            beta_max (float): Description of beta_max.
            beta_flat_epochs (int): Description of beta_flat_epochs.
            kwargs (Any): Description of kwargs.
        """
        super().__init__()

        policy_kwargs = policy_kwargs or {}
        baseline_kwargs = baseline_kwargs or {}

        if policy is None:
            policy = GFACSPolicy(
                env_name=env.name,
                train_with_local_search=train_with_local_search,
                **policy_kwargs,
            )

        self.env = env
        self.policy = policy
        self.train_with_local_search = train_with_local_search

        # TB loss hyperparameters
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.alpha_flat_epochs = alpha_flat_epochs
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.beta_flat_epochs = beta_flat_epochs

        self.current_epoch = 0

    def forward(
        self, td: TensorDict, env: Optional[RL4COEnvBase] = None, phase: str = "train", **kwargs
    ) -> Dict[str, Any]:
        """Forward pass during training/evaluation."""
        env = env or self.env
        return self.policy(td, env, phase=phase, **kwargs)

    @property
    def alpha(self) -> float:
        """Linearly increasing alpha from alpha_min to alpha_max."""
        if not hasattr(self, "trainer") or self.trainer is None or not hasattr(self.trainer, "max_epochs"):
            return self.alpha_min
        return self.alpha_min + (self.alpha_max - self.alpha_min) * min(
            self.current_epoch / (cast(Any, self.trainer).max_epochs - self.alpha_flat_epochs),
            1.0,
        )

    @property
    def beta(self) -> float:
        """Logarithmically increasing beta from beta_min to beta_max."""
        if not hasattr(self, "trainer") or self.trainer is None or not hasattr(self.trainer, "max_epochs"):
            return self.beta_min
        return self.beta_min + (self.beta_max - self.beta_min) * min(
            math.log(self.current_epoch + 1) / math.log(cast(Any, self.trainer).max_epochs - self.beta_flat_epochs),
            1.0,
        )

    def calculate_loss(
        self,
        td: TensorDict,
        batch: TensorDict,
        policy_out: Dict[str, Any],
        reward: Optional[torch.Tensor] = None,
        log_likelihood: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Calculate loss for GFACS algorithm using Trajectory Balance.

        Args:
            td: TensorDict containing the current state of the environment.
            batch: Batch of data. This is used to get the extra loss terms.
            policy_out: Output of the policy network.
            reward: Reward tensor. If None, it is taken from `policy_out`.
            log_likelihood: Log-likelihood tensor. If None, it is taken from `policy_out`.

        Returns:
            Trajectory balance loss.
        """
        reward = policy_out["reward"]
        n_ants = reward.size(1)  # type: ignore
        advantage = reward - reward.mean(dim=1, keepdim=True)  # type: ignore

        if self.train_with_local_search:
            ls_reward = policy_out["ls_reward"]
            ls_advantage = ls_reward - ls_reward.mean(dim=1, keepdim=True)
            weighted_advantage = advantage * (1 - self.alpha) + ls_advantage * self.alpha
        else:
            weighted_advantage = advantage
            ls_advantage = torch.zeros_like(advantage)

        # On-policy loss
        forward_flow = policy_out["log_likelihood"] + policy_out["logZ"].repeat(1, n_ants)
        backward_flow = (
            self.calculate_log_pb_uniform(policy_out["actions"], n_ants) + weighted_advantage.detach() * self.beta
        )
        tb_loss = torch.pow(forward_flow - backward_flow, 2).mean()

        # Off-policy loss
        if self.train_with_local_search:
            ls_forward_flow = policy_out["ls_log_likelihood"] + policy_out["ls_logZ"].repeat(1, n_ants)
            ls_backward_flow = (
                self.calculate_log_pb_uniform(policy_out["ls_actions"], n_ants) + ls_advantage.detach() * self.beta
            )
            ls_tb_loss = torch.pow(ls_forward_flow - ls_backward_flow, 2).mean()
            tb_loss = tb_loss + ls_tb_loss

        return tb_loss

    def calculate_log_pb_uniform(self, actions: torch.Tensor, n_ants: int) -> torch.Tensor:
        """
        Calculate log probability of actions under uniform backward policy.

        Args:
            actions: Action tensor.
            n_ants: Number of ants.

        Returns:
            Log probability tensor.
        """
        if self.env.name == "tsp":
            return torch.tensor(math.log(1 / 2 * actions.size(1)))
        elif self.env.name == "cvrp":
            _a1 = actions.detach().cpu().numpy()
            # shape: (batch, max_tour_length)
            n_nodes = np.count_nonzero(_a1, axis=1)
            _a2 = _a1[:, 1:] - _a1[:, :-1]
            n_routes = np.count_nonzero(_a2, axis=1) - n_nodes
            _a3 = _a1[:, 2:] - _a1[:, :-2]
            n_multinode_routes = np.count_nonzero(_a3, axis=1) - n_nodes
            log_b_p = -scipy.special.gammaln(n_routes + 1) - n_multinode_routes * math.log(2)
            from logic.src.utils.decoding import unbatchify

            return unbatchify(torch.from_numpy(log_b_p).to(actions.device), n_ants)
        elif self.env.name in ("op", "pctsp", "vrpp"):
            return torch.tensor(math.log(1 / 2))
        else:
            raise ValueError(f"Unknown environment: {self.env.name}")
