"""Feed-Forward Convolution Sub-Layer for TGC."""

from torch import nn

from logic.src.models.subnets.modules import (
    ActivationFunction,
    FeedForward,
    GraphConvolution,
)


class FFConvSubLayer(nn.Module):
    """
    Feed-Forward Convolution Sub-Layer.
    """

    def __init__(
        self,
        embed_dim,
        feed_forward_hidden,
        agg,
        activation,
        af_param,
        threshold,
        replacement_value,
        n_params,
        dist_range,
        bias=True,
    ):
        """Initializes the FFConvSubLayer."""
        super(FFConvSubLayer, self).__init__()
        self.conv = GraphConvolution(embed_dim, feed_forward_hidden, agg)
        self.af = ActivationFunction(activation, af_param, threshold, replacement_value, n_params, dist_range)
        self.ff = FeedForward(feed_forward_hidden, embed_dim, bias=bias)

    def forward(self, h, mask=None):
        """Forward pass."""
        h = self.conv(h, mask)
        h = self.af(h)
        return self.ff(h)
