"""Simplified GNN Encoder for NARGNN."""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn

from .gnn_layer import GNNLayer


class SimplifiedGNNEncoder(nn.Module):
    """Simplified GNN encoder for NARGNN."""

    def __init__(self, num_layers: int, embed_dim: int, act_fn: str = "silu", agg_fn: str = "mean"):
        """Initialize Class.

        Args:
            num_layers (int): Description of num_layers.
            embed_dim (int): Description of embed_dim.
            act_fn (str): Description of act_fn.
            agg_fn (str): Description of agg_fn.
        """
        super().__init__()
        self.act_fn = getattr(nn.functional, act_fn)
        self.layers = nn.ModuleList([GNNLayer(embed_dim, act_fn, agg_fn) for _ in range(num_layers)])

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        x = self.act_fn(x)
        edge_attr = self.act_fn(edge_attr)
        for layer in self.layers:
            x, edge_attr = layer(x, edge_index, edge_attr)
        return x, edge_attr
