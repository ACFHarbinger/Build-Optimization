"""Graph Attention Encoder."""

from __future__ import annotations

from typing import Any, Optional

from torch import nn

from logic.src.configs.models.activation_function import ActivationConfig
from logic.src.configs.models.normalization import NormalizationConfig
from logic.src.models.subnets.encoders.common import TransformerEncoderBase

from .gat_multi_head_attention_layer import GATMultiHeadAttentionLayer


class GraphAttentionEncoder(TransformerEncoderBase):
    """
    Graph Attention Encoder with stacked MultiHeadAttentionLayers.

    Supports standard Transformer architecture and Hyper-Networks.
    Inherits from TransformerEncoderBase for common encoder patterns.
    """

    def __init__(
        self,
        n_heads: int,
        embed_dim: int,
        n_layers: int,
        n_sublayers: Optional[int] = None,
        feed_forward_hidden: int = 512,
        norm_config: Optional[NormalizationConfig] = None,
        activation_config: Optional[ActivationConfig] = None,
        dropout_rate: float = 0.1,
        agg: Any = None,
        connection_type: str = "skip",
        expansion_rate: int = 4,
        **kwargs,
    ) -> None:
        """
        Initialize the GraphAttentionEncoder.

        Parameters
        ----------
        n_heads : int
            Number of attention heads.
        embed_dim : int
            Embedding dimension.
        n_layers : int
            Number of encoder layers.
        n_sublayers : Optional[int], default=None
            Number of sublayers (unused, kept for API compatibility).
        feed_forward_hidden : int, default=512
            Hidden dimension for feed-forward layers.
        norm_config : Optional[NormalizationConfig], default=None
            Normalization configuration.
        activation_config : Optional[ActivationConfig], default=None
            Activation function configuration.
        dropout_rate : float, default=0.1
            Dropout probability.
        agg : Any, default=None
            Aggregation method (unused, kept for API compatibility).
        connection_type : str, default="skip"
            Connection type: "skip", "dense", or "hyper".
        expansion_rate : int, default=4
            Expansion factor for hyper-connections.
        **kwargs
            Additional keyword arguments.
        """
        # Initialize base class (handles layer creation, dropout, default configs)
        super(GraphAttentionEncoder, self).__init__(
            n_heads=n_heads,
            embed_dim=embed_dim,
            n_layers=n_layers,
            feed_forward_hidden=feed_forward_hidden,
            norm_config=norm_config,
            activation_config=activation_config,
            dropout_rate=dropout_rate,
            connection_type=connection_type,
            expansion_rate=expansion_rate,
            **kwargs,
        )

        # Store unused parameters for potential future use
        self.n_sublayers = n_sublayers
        self.agg = agg

    def _create_layer(self, layer_idx: int) -> nn.Module:
        """
        Create a single GAT multi-head attention layer.

        Parameters
        ----------
        layer_idx : int
            Index of the layer being created (0 to n_layers-1).

        Returns
        -------
        nn.Module
            GATMultiHeadAttentionLayer instance.
        """
        return GATMultiHeadAttentionLayer(
            n_heads=self.n_heads,
            embed_dim=self.embed_dim,
            feed_forward_hidden=self.feed_forward_hidden,
            norm_config=self.norm_config,
            activation_config=self.activation_config,
            connection_type=self.conn_type,
            expansion_rate=self.expansion_rate,
        )
