"""Graph Attention Convolution Encoder."""

from __future__ import annotations

from typing import Optional

from torch import nn

from logic.src.configs.models.activation_function import ActivationConfig
from logic.src.configs.models.normalization import NormalizationConfig
from logic.src.models.subnets.encoders.common import TransformerEncoderBase

from .attention_convolution_layer import AttentionConvolutionLayer


class GraphAttConvEncoder(TransformerEncoderBase):
    """
    Graph Attention Convolution Encoder using stacked AttentionConvolutionLayers.

    Inherits from TransformerEncoderBase for common encoder patterns.
    """

    def __init__(
        self,
        n_heads: int,
        embed_dim: int,
        n_layers: int,
        n_sublayers: Optional[int] = None,
        feed_forward_hidden: int = 512,
        normalization: str = "batch",
        epsilon_alpha: float = 1e-05,
        learn_affine: bool = True,
        track_stats: bool = False,
        momentum_beta: float = 0.1,
        locresp_k: float = 1.0,
        n_groups: int = 3,
        activation: str = "gelu",
        af_param: float = 1.0,
        threshold: float = 6.0,
        replacement_value: float = 6.0,
        n_params: int = 3,
        uniform_range: Optional[list] = None,
        dropout_rate: float = 0.1,
        aggregate: str = "sum",
        **kwargs,
    ):
        """
        Initialize the GraphAttConvEncoder.

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
        normalization : str, default="batch"
            Normalization type: "batch", "layer", "instance", or "group".
        epsilon_alpha : float, default=1e-05
            Epsilon for normalization stability.
        learn_affine : bool, default=True
            Whether to learn affine parameters in normalization.
        track_stats : bool, default=False
            Whether to track running stats in batch norm.
        momentum_beta : float, default=0.1
            Momentum for batch norm running stats.
        locresp_k : float, default=1.0
            Local response normalization parameter.
        n_groups : int, default=3
            Number of groups for group normalization.
        activation : str, default="gelu"
            Activation function name.
        af_param : float, default=1.0
            Activation function parameter.
        threshold : float, default=6.0
            Activation threshold.
        replacement_value : float, default=6.0
            Activation replacement value.
        n_params : int, default=3
            Number of activation parameters.
        uniform_range : list, default=None
            Uniform range for activation.
        dropout_rate : float, default=0.1
            Dropout probability.
        aggregate : str, default="sum"
            Aggregation method for graph convolutions.
        **kwargs
            Additional keyword arguments.
        """
        # Create config objects from individual parameters
        norm_config = NormalizationConfig(
            norm_type=normalization,
            epsilon=epsilon_alpha,
            learn_affine=learn_affine,
            track_stats=track_stats,
            momentum=momentum_beta,
            k_lrnorm=locresp_k,
            n_groups=n_groups,
        )

        activation_config = ActivationConfig(
            name=activation,
            param=af_param,
            threshold=threshold,
            replacement_value=replacement_value,
            n_params=n_params,
            range=uniform_range if uniform_range is not None else [0.125, 1 / 3],
        )

        # Store GAC-specific parameters BEFORE super().__init__
        # because the base class constructor calls _create_layer which needs these.
        self.aggregate = aggregate
        self.n_sublayers = n_sublayers

        # Initialize base class
        super(GraphAttConvEncoder, self).__init__(
            n_heads=n_heads,
            embed_dim=embed_dim,
            n_layers=n_layers,
            feed_forward_hidden=feed_forward_hidden,
            norm_config=norm_config,
            activation_config=activation_config,
            dropout_rate=dropout_rate,
            **kwargs,
        )

    def _create_layer(self, layer_idx: int) -> nn.Module:
        """
        Create a single AttentionConvolutionLayer.

        Parameters
        ----------
        layer_idx : int
            Index of the layer being created (0 to n_layers-1).

        Returns
        -------
        nn.Module
            AttentionConvolutionLayer instance.
        """
        return AttentionConvolutionLayer(
            self.n_heads,
            self.embed_dim,
            self.feed_forward_hidden,
            self.aggregate,
            self.norm_config.norm_type,
            self.norm_config.epsilon,
            self.norm_config.learn_affine,
            self.norm_config.track_stats,
            self.norm_config.momentum,
            self.norm_config.k_lrnorm,
            self.norm_config.n_groups,
            self.activation_config.name,
            self.activation_config.param,
            self.activation_config.threshold,
            self.activation_config.replacement_value,
            self.activation_config.n_params,
            self.activation_config.range,
        )
