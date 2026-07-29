"""
POMO-style multi-start mean baseline.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
from tensordict import TensorDict

from .base import Baseline


class POMOBaseline(Baseline):
    """
    POMO baseline: mean reward across starts of the SAME instance.
    """

    def __init__(self, **kwargs):
        """Initialize POMOBaseline."""
        super().__init__()

    def eval(self, td: TensorDict, reward: torch.Tensor, env: Optional[Any] = None) -> torch.Tensor:  # type: ignore[override]
        """
        Compute POMO baseline as mean reward across starting points.

        Args:
            td: TensorDict with environment state.
            reward: Reward tensor with shape [batch, num_starts].
            env: Environment (unused).

        Returns:
            torch.Tensor: Mean reward expanded to match input shape.
        """
        # Reward shape: [batch, num_starts]
        if reward.dim() > 1:
            return reward.mean(dim=1, keepdim=True).expand_as(reward)
        return reward.mean()
