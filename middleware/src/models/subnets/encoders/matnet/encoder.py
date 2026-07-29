"""MatNet Encoder."""

from typing import Optional, Tuple

import torch
from torch import nn

from .matnet_encoder_layer import MatNetEncoderLayer


class MatNetEncoder(nn.Module):
    """
    MatNet Encoder.
    Stacks multiple MatNetEncoderLayer.
    """

    def __init__(
        self,
        num_layers: int,
        embed_dim: int,
        n_heads: int,
        feed_forward_hidden: int = 512,
        normalization: str = "instance",
    ):
        """
        Initialize MatNetEncoder.
        """
        super(MatNetEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [MatNetEncoderLayer(embed_dim, n_heads, feed_forward_hidden, normalization) for _ in range(num_layers)]
        )

    def forward(
        self, row_emb: torch.Tensor, col_emb: torch.Tensor, matrix: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        """
        for layer in self.layers:
            row_emb, col_emb = layer(row_emb, col_emb, matrix, mask)
        return row_emb, col_emb
