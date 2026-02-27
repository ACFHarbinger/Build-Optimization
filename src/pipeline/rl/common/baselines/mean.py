"""mean.py module.

Attributes:
    MODULE_VAR (Type): Description of module level variable.

Example:
    >>> import mean
"""

from typing import Any, Optional

import torch
from tensordict import TensorDict

from .base import Baseline


class MeanBaseline(Baseline):
    """
    Simple batch-mean baseline.

    Uses the mean reward of the current batch as the baseline value.
    Zero computational overhead but limited variance reduction.
    Useful as a simple default or for debugging.
    """

    def __init__(self, **kwargs):
        """Initialize MeanBaseline."""
        super().__init__()

    def eval(self, td: TensorDict, reward: torch.Tensor, env: Optional[Any] = None) -> torch.Tensor:  # type: ignore[override]
        """
        Compute baseline as mean of current batch rewards.

        Args:
            td: TensorDict with environment state (unused).
            reward: Current batch rewards.
            env: Environment (unused).

        Returns:
            torch.Tensor: Mean reward expanded to match reward shape.
        """
        return (
            reward.mean(dim=0, keepdim=True).expand_as(reward) if reward.dim() > 1 else reward.mean().expand_as(reward)
        )
