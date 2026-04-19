"""encoder.py module.

Attributes:
    MODULE_VAR (Type): Description of module level variable.

Example:
    >>> import encoder
"""

from __future__ import annotations

from typing import cast

import torch
from tensordict import TensorDict
from torch import nn

from logic.src.models.common.improvement.encoder import ImprovementEncoder
from logic.src.models.subnets.embeddings.positional import pos_init_embedding
from logic.src.models.subnets.modules import MultiHeadAttention, Normalization


class DACTEncoder(ImprovementEncoder):
    """
    DACT Encoder: Combines node features (spatial) and solution features (positional).

    The "Dual Aspect" involves:
    1. Node Aspect: How nodes are located (Graph Convolution/Attention)
    2. Solution Aspect: Where nodes are in the current tour (Positional Encoding)
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 8,
        pos_type: str = "CPE",
        **kwargs,
    ):
        """Initialize DACTEncoder."""
        super().__init__(embed_dim)

        # 1. Node aspect: init embedding + layers
        self.node_init = nn.Linear(2, embed_dim)

        # 2. Solution aspect: positional embedding
        self.pos_embedding = pos_init_embedding(pos_type, embed_dim)

        # 3. Collaborative Transformer layers
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "mha": MultiHeadAttention(num_heads, embed_dim, embed_dim),
                        "norm1": Normalization(embed_dim, "layer"),
                        "ff": nn.Sequential(
                            nn.Linear(embed_dim, embed_dim * 4), nn.ReLU(), nn.Linear(embed_dim * 4, embed_dim)
                        ),
                        "norm2": Normalization(embed_dim, "layer"),
                    }
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, td: TensorDict, **kwargs) -> torch.Tensor:
        """
        Encode problem instance and current solution.

        Args:
            td: TensorDict with 'locs' (customers), 'depot', and 'solution' (tour).

        Returns:
            Embeddings [batch, num_nodes, embed_dim].
        """
        # Combine depot and locs for full nodes
        depot = td["depot"].unsqueeze(1) if td["depot"].dim() == 2 else td["depot"]
        locs = td["locs"]
        nodes = torch.cat([depot, locs], dim=1)

        # Initial node embeddings
        h = self.node_init(nodes)

        # Apply positional encoding based on current solution order
        # solution: [batch, num_nodes] which is the tour (e.g., [0, 5, 2, ...])
        solution = td["solution"]
        num_nodes = solution.size(1)

        # solution contains the node ID at each position
        # We want to find the position index p for each node ID n
        # This is essentially argsort of the solution
        _, pos_idx = solution.sort(dim=1)
        pos_normalized = pos_idx.float() / num_nodes

        # Apply PE
        h = self.pos_embedding(h, pos_normalized)

        # Pass through transformer layers
        for layer_module in self.layers:
            layer = cast(nn.ModuleDict, layer_module)
            # Multi-head attention
            h_attn = layer["mha"](h)
            h = layer["norm1"](h + h_attn)

            # Feed forward
            h_ff = layer["ff"](h)
            h = layer["norm2"](h + h_ff)

        return h
