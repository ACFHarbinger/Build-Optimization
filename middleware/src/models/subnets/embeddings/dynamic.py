"""
Dynamic embeddings for step-dependent updates.

Aligns with rl4co's DynamicEmbedding pattern.
"""

from __future__ import annotations

from typing import Any, Tuple

import torch
from torch import nn


class DynamicEmbedding(nn.Module):
    """
    Dynamic embedding that updates key/values based on state.
    Example: Projecting 'visited' mask or other dynamic node features.
    """

    def __init__(self, embed_dim: int, dynamic_node_dim: int = 1):
        """
        Initialize DynamicEmbedding.

        Args:
            embed_dim: Embedding dimension.
            dynamic_node_dim: Dimension of dynamic features.
        """
        super().__init__()
        self.embed_dim = embed_dim

        # Projection for dynamic node features (e.g., visited mask, remaining waste)
        # We project to 3 * embed_dim to update gl_k, gl_v, logit_k
        self.project_dynamic = nn.Linear(dynamic_node_dim, 3 * embed_dim, bias=False)

    def forward(self, td: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute dynamic updates for glimpse K, V and logit K.

        Checks for 'visited' or 'dynamic_context' in td.

        Args:
            td: TensorDict.

        Returns:
            Tuple of (glimpse_key, glimpse_val, logit_key).
        """
        # Default: check for 'visited' mask as a simple dynamic feature
        # visited: [batch, num_loc]
        dynamic_feat = None

        if "dynamic_context" in td.keys():
            # Explicit dynamic context provided by env
            dynamic_feat = td["dynamic_context"]
        elif "visited" in td.keys():
            # Use visited mask as dynamic feature
            # [batch, num_loc] -> [batch, num_loc, 1]
            dynamic_feat = td["visited"].float().unsqueeze(-1)

        if dynamic_feat is None:
            return torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)

        # Project: [batch, num_loc, 3 * embed_dim]
        out = self.project_dynamic(dynamic_feat)

        # Split
        glimpse_key_dyn, glimpse_val_dyn, logit_key_dyn = out.chunk(3, dim=-1)

        return glimpse_key_dyn, glimpse_val_dyn, logit_key_dyn
