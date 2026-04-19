"""Cyclic positional encoding."""

import math

import torch
from torch import nn


class CyclicPositionalEmbedding(nn.Module):
    """Cyclic positional encoding (Ma et al. 2021)."""

    def __init__(self, embed_dim: int, mean_pooling: bool = True):
        """
        Initialize CyclicPositionalEmbedding.

        Args:
            embed_dim: Embedding dimension.
            mean_pooling: Whether to use mean pooling.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.mean_pooling = mean_pooling
        self.linear = nn.Linear(2, embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Apply cyclic positional encoding.

        Args:
            x: Input tensor (batch, seq_len, embed_dim)
            positions: Normalized positions in [0, 1] (batch, seq_len)

        Returns:
            Tensor with cyclic positional encoding
        """
        # Convert to cyclic coordinates
        angles = 2 * math.pi * positions
        cyclic = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
        pe = self.linear(cyclic)

        if self.mean_pooling:
            pe = pe - pe.mean(dim=1, keepdim=True)

        return x + pe
