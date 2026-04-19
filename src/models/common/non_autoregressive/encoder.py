"""nonautoregressive_encoder.py module.

Attributes:
    MODULE_VAR (Type): Description of module level variable.

Example:
    >>> import nonautoregressive_encoder
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from tensordict import TensorDict
from torch import nn


class NonAutoregressiveEncoder(nn.Module, ABC):
    """
    Base class for non-autoregressive encoders.

    NAR encoders take a problem instance and produce heatmaps
    (log probabilities over edges or nodes) in a single forward pass.
    """

    def __init__(self, embed_dim: int = 128, **kwargs):
        """Initialize NonAutoregressiveEncoder."""
        super().__init__()
        self.embed_dim = embed_dim

    @abstractmethod
    def forward(
        self,
        td: TensorDict,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """
        Compute heatmap logits for the problem instance.

        Args:
            td: TensorDict containing problem instance (locs, wastes, etc.)

        Returns:
            Heatmap tensor of shape [batch, num_nodes, num_nodes] for edge-based,
            or [batch, num_nodes] for node-based.
        """
        raise NotImplementedError
