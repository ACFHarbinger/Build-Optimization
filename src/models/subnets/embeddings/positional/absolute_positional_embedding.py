"""Sinusoidal positional embedding."""

import math

import torch
from torch import nn


class AbsolutePositionalEmbedding(nn.Module):
    """Sinusoidal positional embedding."""

    pe: torch.Tensor

    def __init__(self, embed_dim: int, max_len: int = 5000):
        """
        Initialize AbsolutePositionalEmbedding.

        Args:
            embed_dim: Embedding dimension.
            max_len: Maximum sequence length.
        """
        super().__init__()
        self.embed_dim = embed_dim

        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.

        Args:
            x: Input tensor of shape (batch, seq_len, embed_dim)

        Returns:
            Tensor with positional encoding added
        """
        return x + self.pe[:, : x.size(1)]
