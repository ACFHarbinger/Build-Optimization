"""Transformer Graph Convolution Encoder."""

from torch import nn

from .conv_layer import GraphConvolutionLayer
from .mha_layer import TGCMultiHeadAttentionLayer


class TransGraphConvEncoder(nn.Module):
    """
    Encoder stack of MultiHeadAttentionLayers and GraphConvolutionLayers.
    """

    def __init__(
        self,
        n_heads,
        embed_dim,
        n_layers,
        n_sublayers=None,
        feed_forward_hidden=512,
        normalization="batch",
        epsilon_alpha=1e-05,
        learn_affine=True,
        track_stats=False,
        momentum_beta=0.1,
        locresp_k=1.0,
        n_groups=3,
        activation="gelu",
        af_param=1.0,
        threshold=6.0,
        replacement_value=6.0,
        n_params=3,
        uniform_range=None,
        dropout_rate=0.1,
        agg="mean",
    ):
        """
        Initializes the TransGraphConvEncoder.

        Args:
            n_heads: Number of attention heads.
            embed_dim: Embedding dimension.
            n_layers: Number of attention layers.
            n_sublayers: Number of GCN layers.
            feed_forward_hidden: Hidden dimension.
            normalization: Normalization type.
            epsilon_alpha: Epsilon.
            learn_affine: Learn affine parameters.
            track_stats: Track stats.
            momentum_beta: Momentum.
            locresp_k: K value.
            n_groups: Number of groups.
            activation: Activation function.
            af_param: Activation parameter.
            threshold: Activation threshold.
            replacement_value: Replacement value.
            n_params: Number of parameters.
            uniform_range: Uniform range.
            dropout_rate: Dropout rate.
            agg: Aggregation method.
        """
        if uniform_range is None:
            uniform_range = [0.125, 1 / 3]
        super(TransGraphConvEncoder, self).__init__()
        layers = [
            TGCMultiHeadAttentionLayer(
                n_heads,
                embed_dim,
                feed_forward_hidden,
                normalization,
                epsilon_alpha,
                learn_affine,
                track_stats,
                momentum_beta,
                locresp_k,
                n_groups,
                activation,
                af_param,
                threshold,
                replacement_value,
                n_params,
                uniform_range,
            )
            for _ in range(n_layers)
        ] + [
            GraphConvolutionLayer(
                embed_dim,
                feed_forward_hidden,
                agg,
                normalization,
                epsilon_alpha,
                learn_affine,
                track_stats,
                momentum_beta,
                locresp_k,
                n_groups,
                activation,
                af_param,
                threshold,
                replacement_value,
                n_params,
                uniform_range,
            )
            for _ in range(n_sublayers or 0)
        ]
        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, edges):
        """
        Forward pass.

        Args:
            x: Input embeddings.
            edges: Adjacency/Mask.

        Returns:
            Encoded embeddings.
        """
        for layer in self.layers:
            x = layer(x, edges)
        return self.dropout(x)  # (batch_size, graph_size, embed_dim)
