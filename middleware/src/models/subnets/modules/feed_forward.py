"""
Feed-Forward network module.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn


class FeedForward(nn.Module):
    """
    Standard Feed-Forward Neural Network (MLP) block.

    Consists of two linear transformations with an activation function in between.
    Often used in Transformer architectures.
    """

    def __init__(self, input_dim: int, output_dim: int, bias: bool = True) -> None:
        """
        Initializes the feed-forward layer.

        Args:
            input_dim: Input dimension.
            output_dim: Output dimension.
            bias: Whether to include a bias term in the linear layer.
        """
        super(FeedForward, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)
        self.init_parameters()

    def init_parameters(self) -> None:
        """Initializes the parameters using uniform distribution."""
        for param in self.parameters():
            if param.dim() > 0:
                stdv: float = 1.0 / math.sqrt(param.size(-1))
                param.data.uniform_(-stdv, stdv)

    def forward(self, input: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Applies the feed-forward network to the input.

        Args:
            input: Input tensor.
            mask: Optional mask (not used in this basic implementation).

        Returns:
            Output tensor.
        """
        return self.linear(input)
