"""
VRPP Embedding module.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict
from torch import nn


class VRPPInitEmbedding(nn.Module):
    """Initial embedding for VRPP problems."""

    def __init__(self, embed_dim: int = 128):
        """
        Initialize VRPPInitEmbedding.

        Args:
            embed_dim: Embedding dimension.
        """
        super().__init__()
        # Node features: x, y, waste
        self.node_embed = nn.Linear(3, embed_dim)
        # Depot features: x, y
        self.depot_embed = nn.Linear(2, embed_dim)

    def forward(self, td: TensorDict) -> torch.Tensor:
        """
        Embed VRPP instance.

        Args:
            td: TensorDict with keys: locs, depot, waste

        Returns:
            Node embeddings [batch, num_nodes, embed_dim]
        """
        locs = td["locs"]  # [batch, num_nodes, 2]
        waste = td.get("waste")

        if waste is None:
            # Fallback if missing
            waste = torch.zeros(td.batch_size[0], locs.size(1), device=td.device)

        # Combine location and waste for non-depot nodes
        node_features = torch.cat([locs, waste.unsqueeze(-1)], dim=-1)
        embeddings = self.node_embed(node_features)

        # Special embedding for depot
        depot_emb = self.depot_embed(td["depot"])
        embeddings[:, 0] = depot_emb

        return embeddings
