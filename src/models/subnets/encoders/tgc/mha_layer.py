"""Multi-Head Attention Layer for TGC."""

from torch import nn

from logic.src.models.subnets.modules import (
    MultiHeadAttention,
    Normalization,
    SkipConnection,
)

from .ff_sublayer import TGCFeedForwardSubLayer


class TGCMultiHeadAttentionLayer(nn.Module):
    """
    Multi-Head Attention Layer with Normalization and Feed-Forward.
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
    ):
        """Initializes the TGCMultiHeadAttentionLayer."""
        super(TGCMultiHeadAttentionLayer, self).__init__()
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
        self.ff = SkipConnection(
            TGCFeedForwardSubLayer(
                embed_dim,
                feed_forward_hidden,
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

    def forward(self, h, mask=None):
        """Forward pass with skip connections and normalization."""
        h = self.att(h)
        h = self.norm1(h)
        h = self.ff(h)
        return self.norm2(h)
