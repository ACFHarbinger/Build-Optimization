"""
Neural critic-based baselines.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
from tensordict import TensorDict
from torch import nn

from logic.src.utils.functions.rl import ensure_tensordict

from .base import Baseline


class CriticBaseline(Baseline):
    """Learned critic baseline."""

    def __init__(self, critic: Optional[nn.Module] = None, **kwargs):
        """
        Initialize CriticBaseline.

        Args:
            critic: Critic neural network module.
        """
        super().__init__()
        self.critic = critic

    def eval(self, td: TensorDict, reward: torch.Tensor, env: Optional[Any] = None) -> torch.Tensor:  # type: ignore[override]
        """
        Compute baseline value using learned critic.

        Args:
            td: TensorDict with environment state.
            reward: Current batch rewards (used for shape if critic is None).
            env: Environment (unused).

        Returns:
            torch.Tensor: Critic value predictions.
        """
        if self.critic is None:
            return torch.zeros_like(reward)

        td = ensure_tensordict(td, next(iter(self.critic.parameters())).device)
        return self.critic(td).squeeze(-1)

    def get_learnable_parameters(self) -> list:
        """Get learnable parameters for the critic network."""
        return list(self.critic.parameters()) if self.critic is not None else []

    def state_dict(self, *args, **kwargs):
        """Compatibility state_dict."""
        sd = super().state_dict(*args, **kwargs)
        if self.critic is not None:
            sd["critic"] = self.critic.state_dict()
        return sd
