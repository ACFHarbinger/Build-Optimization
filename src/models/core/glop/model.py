"""
GLOP Model.

REINFORCE training for Global-Local Optimization Policy.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from tensordict import TensorDict
from torch import nn

from logic.src.envs.base.base import RL4COEnvBase

from .policy import GLOPPolicy


class GLOP(nn.Module):
    """
    GLOP Model: Global-Local Optimization with REINFORCE.

    Trains a NAR partitioning policy using REINFORCE with mean baseline.
    The local solvers are typically non-differentiable heuristics.

    Reference:
        Ye et al. "GLOP: Learning Global Partition and Local Construction
        for Solving Large-scale Routing Problems in Real-time" (2023)
    """

    def __init__(
        self,
        env: RL4COEnvBase,
        policy: Optional[GLOPPolicy] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        baseline: str = "mean",
        **kwargs,
    ) -> None:
        """
        Initialize GLOP model.

        Args:
            env: Environment for training.
            policy: Pre-built policy or None to create default.
            policy_kwargs: Arguments for policy construction.
            baseline: Baseline type (only "mean" supported).
        """
        super().__init__()

        policy_kwargs = policy_kwargs or {}

        self.env = env
        self.baseline = baseline

        if policy is None:
            self.policy = GLOPPolicy(env_name=env.name, **policy_kwargs)
        else:
            self.policy = policy

        self.n_samples = self.policy.n_samples

    def forward(
        self,
        td: TensorDict,
        env: Optional[RL4COEnvBase] = None,
        phase: str = "train",
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Forward pass through GLOP policy.

        Args:
            td: Problem instance.
            env: Environment.
            phase: Training phase.

        Returns:
            Policy output dictionary.
        """
        if env is None:
            env = self.env
        return self.policy(td, env, phase=phase, **kwargs)

    def shared_step(
        self,
        batch: TensorDict,
        batch_idx: int,
        phase: str,
    ) -> Dict[str, Any]:
        """
        Shared step for training/validation/testing.

        Args:
            batch: Batch of problem instances.
            batch_idx: Batch index.
            phase: Current phase.

        Returns:
            Dictionary with loss and metrics.
        """
        td = self.env.reset(batch)

        # Forward pass
        out = self.policy(
            td=td,
            env=self.env,
            phase=phase,
            return_actions=True,
        )

        # Reshape reward for n_samples
        from logic.src.utils.decoding import unbatchify

        reward = unbatchify(out["reward"], self.n_samples)
        max_reward, max_idxs = reward.max(dim=-1)
        out["max_reward"] = max_reward

        # Compute loss for training
        if phase == "train":
            assert self.n_samples > 1, "n_samples must be > 1 for training"
            log_likelihood = unbatchify(out["log_likelihood"], self.n_samples)

            # Mean baseline advantage
            advantage = reward - reward.mean(dim=-1, keepdim=True)
            loss = -(advantage * log_likelihood).mean()
            out["loss"] = loss

        return out

    def calculate_loss(
        self,
        td: TensorDict,
        batch: TensorDict,
        policy_out: Dict[str, Any],
        reward: Optional[torch.Tensor] = None,
        log_likelihood: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Calculate REINFORCE loss with mean baseline.

        Args:
            td: Problem instance.
            batch: Batch data.
            policy_out: Output from policy forward pass.
            reward: Override reward.
            log_likelihood: Override log_likelihood.

        Returns:
            Updated policy_out with loss.
        """
        reward = reward if reward is not None else policy_out["reward"]
        log_likelihood = log_likelihood if log_likelihood is not None else policy_out["log_likelihood"]

        # Reshape for multi-sample
        from logic.src.utils.decoding import unbatchify

        reward = unbatchify(reward, self.n_samples)
        log_likelihood = unbatchify(log_likelihood, self.n_samples)

        # Mean baseline
        advantage = reward - reward.mean(dim=-1, keepdim=True)
        loss = -(advantage * log_likelihood).mean()

        policy_out["loss"] = loss
        policy_out["max_reward"] = reward.max(dim=-1).values.mean()

        return policy_out
