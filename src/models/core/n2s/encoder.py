"""encoder.py module.

Attributes:
    MODULE_VAR (Type): Description of module level variable.

Example:
    >>> import encoder
"""

from __future__ import annotations

import torch
from tensordict import TensorDict
from torch import nn

from logic.src.models.common.improvement.encoder import ImprovementEncoder
from logic.src.models.subnets.modules import MultiHeadAttention, Normalization


class N2SEncoder(ImprovementEncoder):
    """
    N2S Encoder: A lightweight transformer with neighborhood masking.

    Attends only to K-nearest neighbors to improve efficiency and focus
    on local search moves.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        k_neighbors: int = 20,
        **kwargs,
    ):
        """
        Initialize N2SEncoder.

        Args:
            embed_dim: Embedding dimension.
            num_heads: Number of attention heads.
            k_neighbors: Number of nearest neighbors to attend to.
            **kwargs: Unused arguments.
        """
        super().__init__(embed_dim=embed_dim)
        self.k_neighbors = k_neighbors
        self.num_heads = num_heads

        self.mha = MultiHeadAttention(num_heads, embed_dim, embed_dim)
        self.norm = Normalization(embed_dim, "layer")
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def _get_neighborhood_mask(self, td: TensorDict) -> torch.Tensor:
        """
        Compute mask for K-nearest neighbors based on spatial distance.
        """
        # locs: [bs, n, 2]
        depot = td["depot"].unsqueeze(1)
        customers = td["locs"]
        locs = torch.cat([depot, customers], dim=1)

        # dist: [bs, n, n]
        dist = torch.cdist(locs, locs)

        # Get indices of K-nearest neighbors
        # (self is always included in nearest)
        _, topk_indices = dist.topk(self.k_neighbors + 1, largest=False, sorted=False)

        # Create mask: True for excluded neighbors
        bs, n, _ = dist.shape
        mask = torch.ones(bs, n, n, device=dist.device, dtype=torch.bool)
        mask.scatter_(2, topk_indices, False)

        return mask

    def forward(self, td: TensorDict, **kwargs) -> torch.Tensor:
        """
        Encode nodes with neighborhood-constrained attention.
        """
        # Initial embedding (coords)
        depot = td["depot"].unsqueeze(1)
        customers = td["locs"]
        h = torch.cat([depot, customers], dim=1)  # [bs, n, 2]

        # Linear projection if needed (h is [bs, n, 2], we want embed_dim)
        # Assuming we have an init_embedding or simple linear
        if not hasattr(self, "init_proj"):
            self.init_proj = nn.Linear(2, self.embed_dim).to(h.device)

        h = self.init_proj(h)

        # Neighborhood mask
        mask = self._get_neighborhood_mask(td)

        # Transformer layer
        attn_out = self.mha(h, mask=mask)
        h = self.norm(h + attn_out)
        h = h + self.ff(h)

        return h
