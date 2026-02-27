"""none.py module.

Attributes:
    MODULE_VAR (Type): Description of module level variable.

Example:
    >>> import none
"""

from typing import Any, Optional

import torch
from tensordict import TensorDict

from .base import Baseline


class NoBaseline(Baseline):
    """No baseline (vanilla REINFORCE)."""

    def __init__(self, **kwargs):
        """Initialize NoBaseline."""
        super().__init__()

    def eval(self, td: TensorDict, reward: torch.Tensor, env: Optional[Any] = None) -> torch.Tensor:  # type: ignore[override]
        """
        Return zero baseline (no variance reduction).

        Args:
            td: TensorDict with environment state (unused).
            reward: Current batch rewards.
            env: Environment (unused).

        Returns:
            torch.Tensor: Zeros matching the reward shape.
        """
        return torch.zeros_like(reward)
