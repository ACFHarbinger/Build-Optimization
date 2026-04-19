"""nonautoregressive_decoder.py module.

Attributes:
    MODULE_VAR (Type): Description of module level variable.

Example:
    >>> import nonautoregressive_decoder
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple

import torch
from tensordict import TensorDict
from torch import nn

from logic.src.envs.base.base import RL4COEnvBase


class NonAutoregressiveDecoder(nn.Module, ABC):
    """
    Base class for non-autoregressive decoders.

    NAR decoders construct solutions from heatmaps using various methods
    like Ant Colony Optimization, greedy decoding, or sampling.
    """

    def __init__(self, **kwargs):
        """Initialize NonAutoregressiveDecoder."""
        super().__init__()

    @abstractmethod
    def forward(
        self,
        td: TensorDict,
        heatmap: torch.Tensor,
        env: RL4COEnvBase,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Produce logits and mask for the current step.

        Args:
            td: TensorDict containing current state.
            heatmap: Pre-computed heatmap from encoder.
            env: Environment.

        Returns:
            Tuple of (logits, mask).
        """
        raise NotImplementedError

    def construct(
        self,
        td: TensorDict,
        heatmap: torch.Tensor,
        env: RL4COEnvBase,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Full construction of solutions from heatmaps.
        Default implementation returns empty dict, subclasses should override.
        """
        return {}
