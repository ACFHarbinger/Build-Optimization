"""Attention Gated Convolution Layer for GGAC Encoder."""

from __future__ import annotations

from torch import nn

from logic.src.models.subnets.modules import (
    ActivationFunction,
    FeedForward,
    GatedGraphConvolution,
    MultiHeadAttention,
    Normalization,
    SkipConnection,
)


class AttentionGatedConvolutionLayer(nn.Module):
    """
    Combined Layer with MHA, Gated GCN, and Feed-Forward Network.
    """

    def __init__(
        self,
        n_heads,
        embed_dim,
        feed_forward_hidden,
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
        gated=True,
        agg="sum",
        bias=True,
    ):
        """Initializes the AttentionGatedConvolutionLayer."""
        super(AttentionGatedConvolutionLayer, self).__init__()

        # 1. Multi-Head Attention (Global Context)
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

        # 2. Gated Graph Convolution (Local Structure + Edge Updates)
        self.gated_gcn = GatedGraphConvolution(
            hidden_dim=embed_dim,
            aggregation=agg,
            norm=normalization,
            activation=activation,
            learn_affine=learn_affine,
            gated=gated,
            bias=bias,
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

        # 3. Feed Forward (Node-wise processing)
        self.ff = (
            SkipConnection(
                nn.Sequential(
                    FeedForward(embed_dim, feed_forward_hidden, bias=bias),
                    ActivationFunction(
                        activation,
                        af_param,
                        threshold,
                        replacement_value,
                        n_params,
                        uniform_range,
                    ),
                    FeedForward(feed_forward_hidden, embed_dim, bias=bias),
                )
            )
            if feed_forward_hidden > 0
            else SkipConnection(FeedForward(embed_dim, embed_dim))
        )

        self.norm3 = Normalization(
            embed_dim,
            normalization,
            epsilon_alpha,
            learn_affine,
            track_stats,
            mbeta,
            n_groups,
            lr_k,
        )

    def forward(self, h, e, mask=None):
        """Forward pass."""
        # 1. MHA
        h = self.att(h, mask=mask)
        h = self.norm1(h)

        # 2. Gated GCN
        h_in = h
        e_in = e

        h_gcn, e_gcn = self.gated_gcn(h, e, mask)

        h = h_in + h_gcn  # Residual for Node
        e = e_in + e_gcn  # Residual for Edge

        h = self.norm2(h)

        # 3. FF
        h = self.ff(h)
        h = self.norm3(h)

        return h, e
