"""Multi-Head Attention Layer for GAT Encoder."""

from __future__ import annotations

from typing import Optional

from logic.src.configs.models.activation_function import ActivationConfig
from logic.src.configs.models.normalization import NormalizationConfig
from logic.src.models.subnets.encoders.common import MultiHeadAttentionLayerBase


class GATMultiHeadAttentionLayer(MultiHeadAttentionLayerBase):
    """
    Single layer of the Graph Attention Encoder.

    Inherits from MultiHeadAttentionLayerBase, which provides:
    - Multi-head attention with configurable connections (skip/dense/hyper)
    - Feed-forward network with configurable activation
    - Layer normalization
    - Automatic 4D tensor handling for hyper-connections

    This class is a direct application of the base pattern with no customization needed.

    Parameters
    ----------
    n_heads : int
        Number of attention heads.
    embed_dim : int
        Embedding dimension.
    feed_forward_hidden : int
        Hidden dimension for feed-forward network.
    norm_config : Optional[NormalizationConfig], default=None
        Normalization configuration.
    activation_config : Optional[ActivationConfig], default=None
        Activation function configuration.
    connection_type : str, default="skip"
        Connection type: "skip", "dense", or "hyper".
    expansion_rate : int, default=4
        Expansion factor for hyper-connections.
    **kwargs
        Additional keyword arguments.
    """

    def __init__(
        self,
        n_heads: int,
        embed_dim: int,
        feed_forward_hidden: int,
        norm_config: Optional[NormalizationConfig] = None,
        activation_config: Optional[ActivationConfig] = None,
        connection_type: str = "skip",
        expansion_rate: int = 4,
        **kwargs,
    ) -> None:
        """Initialize the GATMultiHeadAttentionLayer."""
        # Simply delegate to base class - all logic is inherited
        super(GATMultiHeadAttentionLayer, self).__init__(
            n_heads=n_heads,
            embed_dim=embed_dim,
            feed_forward_hidden=feed_forward_hidden,
            norm_config=norm_config,
            activation_config=activation_config,
            connection_type=connection_type,
            expansion_rate=expansion_rate,
            **kwargs,
        )
