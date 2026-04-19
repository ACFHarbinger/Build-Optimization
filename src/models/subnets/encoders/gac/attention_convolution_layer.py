"""Attention Convolution Layer for GAC Encoder."""

from __future__ import annotations

from torch import nn

from logic.src.models.subnets.modules import (
    MultiHeadAttention,
    Normalization,
    SkipConnection,
)

from .ff_conv_sublayer import FFConvSubLayer


class AttentionConvolutionLayer(nn.Module):
    """
    Combined Attention and Convolution Layer.
    """

    def __init__(
        self,
        n_heads,
        embed_dim,
        feed_forward_hidden,
        agg,
        normalization,
        epsilon_alpha,
        learn_affine,
        track_stats,
        mbeta,
        lr_k,
        n_groups,
        activation,
        af_param,
        threshold,
        replacement_value,
        n_params,
        uniform_range,
    ):
        """Initializes the Attention Convolution Layer."""
        super(AttentionConvolutionLayer, self).__init__()
        self.att = SkipConnection(MultiHeadAttention(n_heads, input_dim=embed_dim, embed_dim=embed_dim))
        self.norm1 = Normalization(
            embed_dim,
            normalization,
            epsilon_alpha,
            learn_affine,
            track_stats,
            mbeta,
            n_groups,
            lr_k,
        )
        self.ff_conv = SkipConnection(
            FFConvSubLayer(
                embed_dim,
                feed_forward_hidden,
                agg,
                activation,
                af_param,
                threshold,
                replacement_value,
                n_params,
                uniform_range,
            )
        )
        self.norm2 = Normalization(
            embed_dim,
            normalization,
            epsilon_alpha,
            learn_affine,
            track_stats,
            mbeta,
            n_groups,
            lr_k,
        )

    def forward(self, h, edges=None, mask=None):
        """Forward pass."""
        h = self.att(h, mask=mask)
        h = self.norm1(h)
        # Use edges if provided (typical for graph conv), otherwise fallback to mask
        adj = edges if edges is not None else mask
        h = self.ff_conv(h, mask=adj)
        return self.norm2(h)
