"""MPNN Encoder stack."""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn

from .mpnn_layer import MessagePassingLayer


class MPNNEncoder(nn.Module):
    """
    Encoder stack using MessagePassingLayer.
    """

    def __init__(
        self,
        num_layers: int,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 64,
        aggr: str = "add",
        norm: str = "batch",
    ):
        """
        Initialize MPNNEncoder.
        """
        super().__init__()

        # Project dimensions if needed
        self.input_proj = nn.Linear(node_dim, hidden_dim) if node_dim != hidden_dim else nn.Identity()
        self.edge_proj = nn.Linear(edge_dim, hidden_dim) if edge_dim != hidden_dim else nn.Identity()

        self.layers = nn.ModuleList(
            [
                MessagePassingLayer(
                    node_dim=hidden_dim, edge_dim=hidden_dim, hidden_dim=hidden_dim, aggr=aggr, norm=norm
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        """
        x = self.input_proj(x)
        edge_attr = self.edge_proj(edge_attr)

        for layer in self.layers:
            x_new, edge_attr_new = layer(x, edge_index, edge_attr)
            x = x + x_new  # Residual
            edge_attr = edge_attr + edge_attr_new  # Residual

        return x, edge_attr
