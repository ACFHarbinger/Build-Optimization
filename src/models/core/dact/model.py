"""
DACT Model implementation.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from tensordict import TensorDict
from torch import nn

from logic.src.envs.base.base import RL4COEnvBase

from .policy import DACTPolicy


class DACT(nn.Module):
    """
    DACT: Dual Aspect Collaborative Transformer for iterative improvement.

    This model wraps a DACTPolicy and provides methods for loss calculation
    and shared execution steps.
    """

    def __init__(
        self,
        env: RL4COEnvBase,
        policy: Optional[DACTPolicy] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        baseline: str = "rollout",
        **kwargs,
    ):
        """Initialize DACT model."""
        super().__init__()
        self.env = env
        self.baseline = baseline

        policy_kwargs = policy_kwargs or {}
        if policy is None:
            self.policy = DACTPolicy(env_name=env.name, **policy_kwargs)
        else:
            self.policy = policy

    def forward(
        self,
        td: TensorDict,
        env: Optional[RL4COEnvBase] = None,
        phase: str = "test",
        **kwargs,
    ) -> Dict[str, Any]:
        """Forward pass through policy."""
        if env is None:
            env = self.env
        return self.policy(td, env, phase=phase, **kwargs)

    def calculate_loss(
        self,
        td: TensorDict,
        batch: TensorDict,
        policy_out: Dict[str, Any],
        reward: Optional[torch.Tensor] = None,
        log_likelihood: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Calculate loss for iterative improvement.
        Typically uses REINFORCE with baseline on the final reward.
        """
        reward = reward if reward is not None else policy_out["reward"]
        log_likelihood = log_likelihood if log_likelihood is not None else policy_out["log_likelihood"]

        # Simple REINFORCE with mean baseline for now
        # Improvement models often use more complex baselines or PPO
        advantage = reward - reward.mean()
        loss = -(advantage * log_likelihood).mean()

        policy_out["loss"] = loss
        return policy_out

    def shared_step(
        self,
        batch: TensorDict,
        batch_idx: int,
        phase: str,
    ) -> Dict[str, Any]:
        """Step used by training pipelines."""
        td = self.env.reset(batch)
        out = self.policy(td, self.env, phase=phase)

        if phase == "train":
            out = self.calculate_loss(td, batch, out)

        return out
