"""
DeepACO Encoder: GNN-based heatmap predictor.

Predicts edge log-probabilities (heatmap) for guiding ant colony construction.
"""

from __future__ import annotations

from typing import Tuple, Union

import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch import nn

from logic.src.models.common.non_autoregressive.encoder import NonAutoregressiveEncoder


class DeepACOEncoder(NonAutoregressiveEncoder):
    """
    GNN-based encoder for DeepACO.

    Predicts edge heatmap (log probabilities) from problem instance.
    Uses message passing to aggregate neighbor information.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 8,
        feedforward_dim: int = 512,
        dropout: float = 0.0,
        input_dim: int = 2,
        **kwargs,
    ):
        """
        Initialize DeepACOEncoder.

        Args:
            embed_dim: Embedding dimension.
            num_layers: Number of GNN layers.
            num_heads: Number of attention heads.
            feedforward_dim: Feed-forward hidden dimension.
            dropout: Dropout rate.
            input_dim: Input feature dimension (default 2 for 2D coordinates).
        """
        super().__init__(embed_dim=embed_dim, **kwargs)
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.input_dim = input_dim

        # Initial embedding
        self.init_embed = nn.Linear(input_dim, embed_dim)

        # GNN layers (using simple self-attention as message passing)
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=num_heads,
                    dim_feedforward=feedforward_dim,
                    dropout=dropout,
                    batch_first=True,
                )
                for _ in range(num_layers)
            ]
        )

        # Edge predictor: maps node pair embeddings to edge logits
        self.edge_predictor = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(  # type: ignore[override]
        self,
        td: TensorDict,
        return_embeddings: bool = False,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute edge heatmap logits.

        Args:
            td: TensorDict with 'locs' [batch, num_nodes, 2].
            return_embeddings: If True, also return node embeddings.

        Returns:
            Heatmap tensor [batch, num_nodes, num_nodes] with edge log probabilities.
        """
        locs = td["locs"]  # [batch, num_nodes, input_dim]
        batch_size, num_nodes, _ = locs.shape

        # Initial node embeddings
        x = self.init_embed(locs)  # [batch, num_nodes, embed_dim]

        # Apply GNN layers
        for layer in self.layers:
            x = layer(x)

        # Compute edge logits via outer product + MLP
        # Create all pairs: [batch, num_nodes, num_nodes, 2*embed_dim]
        x_i = x.unsqueeze(2).expand(-1, -1, num_nodes, -1)  # [batch, n, n, d]
        x_j = x.unsqueeze(1).expand(-1, num_nodes, -1, -1)  # [batch, n, n, d]
        edge_features = torch.cat([x_i, x_j], dim=-1)  # [batch, n, n, 2d]

        # Predict edge logits
        heatmap = self.edge_predictor(edge_features).squeeze(-1)  # [batch, n, n]

        # Apply log-softmax over edges (row-wise normalization)
        heatmap = F.log_softmax(heatmap, dim=-1)

        if return_embeddings:
            return heatmap, x
        return heatmap
