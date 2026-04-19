"""
Graph Attention Decoder module.
"""

from typing import Optional

import torch
from torch import nn

from logic.src.configs.models.activation_function import ActivationConfig
from logic.src.configs.models.normalization import NormalizationConfig

from .multi_head_attention_layer import MultiHeadAttentionLayer


class GraphAttentionDecoder(nn.Module):
    """
    Decoder composed of stacked MultiHeadAttentionLayers.
    Projects the final output to logits/probabilities.
    """

    def __init__(
        self,
        n_heads: int,
        embed_dim: int,
        n_layers: int,
        feed_forward_hidden: int = 512,
        norm_config: Optional[NormalizationConfig] = None,
        activation_config: Optional[ActivationConfig] = None,
        dropout_rate: float = 0.1,
        **kwargs,
    ):
        """
        Initializes the GraphAttentionDecoder.
        """
        super(GraphAttentionDecoder, self).__init__()

        if norm_config is None:
            norm_config = NormalizationConfig()
        if activation_config is None:
            activation_config = ActivationConfig()

        self.layers = nn.ModuleList(
            [
                MultiHeadAttentionLayer(
                    n_heads,
                    embed_dim,
                    feed_forward_hidden,
                    norm_config,
                    activation_config,
                )
                for _ in range(n_layers)
            ]
        )
        self.projection = nn.Linear(embed_dim, 2)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, q, h=None, mask=None):
        """
        Forward pass.

        Args:
            q: Query embeddings.
            h: Node embeddings (keys/values). If None, uses q (self-attention).
            mask: Attention mask.

        Returns:
            Softmax probabilities over the sequence.
        """
        if h is None:
            h = q  # compute self-attention

        for layer in self.layers:
            h = layer(q, h, mask)

        out = self.projection(self.dropout(h))
        return torch.softmax(out, dim=-1)
