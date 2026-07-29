"""
WCVRP Embedding module.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict
from torch import nn


class WCVRPInitEmbedding(nn.Module):
    """Initial embedding for WCVRP problems."""

    def __init__(self, embed_dim: int = 128):
        """
        Initialize WCVRPInitEmbedding.

        Args:
            embed_dim: Embedding dimension.
        """
        super().__init__()
        # Node features: x, y, waste (fill_level)
        self.node_embed = nn.Linear(3, embed_dim)
        self.depot_embed = nn.Linear(2, embed_dim)

    def forward(self, td: TensorDict) -> torch.Tensor:
        """
        Forward pass for WCVRP embedding.

        Args:
            td: TensorDict containing 'locs', 'waste' (fill levels), 'depot'.

        Returns:
            Embeddings tensor [batch, num_nodes, embed_dim].
        """
        locs = td["locs"]
        waste = td.get("waste")  # Fill levels

        if waste is None:
            waste = torch.zeros(td.batch_size[0], locs.size(1), device=td.device)

        node_features = torch.cat([locs, waste.unsqueeze(-1)], dim=-1)
        embeddings = self.node_embed(node_features)

        depot_emb = self.depot_embed(td["depot"])
        embeddings[:, 0] = depot_emb

        return embeddings
