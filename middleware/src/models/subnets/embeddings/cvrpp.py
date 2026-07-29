"""
CVRPP Embedding module.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict
from torch import nn


class CVRPPInitEmbedding(nn.Module):
    """Initial embedding for CVRPP (capacitated VRPP)."""

    def __init__(self, embed_dim: int = 128):
        """
        Initialize CVRPPInitEmbedding.

        Args:
            embed_dim: Embedding dimension.
        """
        super().__init__()
        # Node features: x, y, waste
        self.node_embed = nn.Linear(3, embed_dim)
        self.depot_embed = nn.Linear(2, embed_dim)

    def forward(self, td: TensorDict) -> torch.Tensor:
        """
        Forward pass for CVRPP embedding.

        Args:
            td: TensorDict containing 'locs', 'waste', 'depot'.

        Returns:
            Embeddings tensor [batch, num_nodes, embed_dim].
        """
        locs = td["locs"]
        waste = td.get("waste", torch.zeros(td.batch_size[0], locs.size(1), device=td.device))

        node_features = torch.cat([locs, waste.unsqueeze(-1)], dim=-1)
        embeddings = self.node_embed(node_features)

        depot_emb = self.depot_embed(td["depot"])
        embeddings[:, 0] = depot_emb

        return embeddings
