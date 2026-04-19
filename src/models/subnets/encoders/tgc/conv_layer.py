"""Graph Convolution Layer for TGC."""

from torch import nn

from logic.src.models.subnets.modules import Normalization, SkipConnection

from .conv_sublayer import FFConvSubLayer


class GraphConvolutionLayer(nn.Module):
    """
    Graph Convolution Layer with Normalization.
    """

    def __init__(
        self,
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
        """Initializes the GraphConvolutionLayer."""
        super(GraphConvolutionLayer, self).__init__()
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
        self.norm = Normalization(
            embed_dim,
            normalization,
            epsilon_alpha,
            learn_affine,
            track_stats,
            mbeta,
            n_groups,
            lr_k,
        )

    def forward(self, h, mask):
        """Forward pass."""
        h = self.ff_conv(h, mask=mask)
        return self.norm(h)
