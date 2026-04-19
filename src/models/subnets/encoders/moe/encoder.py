"""MoE Graph Attention Encoder."""

from __future__ import annotations

from typing import Optional

from torch import nn

from logic.src.configs.models.activation_function import ActivationConfig
from logic.src.configs.models.normalization import NormalizationConfig
from logic.src.models.subnets.encoders.common import TransformerEncoderBase

from .moe_multi_head_attention_layer import MoEMultiHeadAttentionLayer


class MoEGraphAttentionEncoder(TransformerEncoderBase):
    """
    MoE (Mixture of Experts) Graph Attention Encoder.

    Extends TransformerEncoderBase with MoE-specific multi-head attention layers.
    Each layer uses a mixture of expert feed-forward networks for improved
    capacity and specialization.

    Inherits from TransformerEncoderBase, which provides:
    - Layer stacking and management
    - Hyper-connection support (skip/dense/hyper)
    - Dropout application
    - Standard forward pass

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
        Hidden dimension for each expert's feed-forward network.
    normalization : str, default="batch"
        Normalization type: "batch", "layer", "instance", or "group".
    norm_eps_alpha : float, default=1e-05
        Epsilon for normalization stability.
    norm_learn_affine : bool, default=True
        Whether to learn affine parameters in normalization.
    norm_track_stats : bool, default=False
        Whether to track running stats in batch norm.
    norm_momentum_beta : float, default=0.1
        Momentum for batch norm running stats.
    lrnorm_k : float, default=1.0
        Local response normalization parameter.
    gnorm_groups : int, default=3
        Number of groups for group normalization.
    activation_function : str, default="gelu"
        Activation function name.
    af_param : float, default=1.0
        Activation function parameter.
    af_threshold : float, default=6.0
        Activation threshold.
    af_replacement_value : float, default=6.0
        Activation replacement value.
    af_num_params : int, default=3
        Number of activation parameters.
    af_uniform_range : list, default=[0.125, 1/3]
        Uniform range for activation.
    dropout_rate : float, default=0.1
        Dropout probability.
    agg : Any, default=None
        Aggregation method (unused, kept for API compatibility).
    connection_type : str, default="skip"
        Connection type: "skip", "dense", or "hyper".
    expansion_rate : int, default=4
        Expansion factor for hyper-connections.
    num_experts : int, default=4
        Number of expert networks in MoE.
    k : int, default=2
        Number of experts to activate per input (top-k routing).
    noisy_gating : bool, default=True
        Whether to use noisy gating for exploration.
    **kwargs
        Additional keyword arguments.
    """

    def __init__(
        self,
        n_heads,
        embed_dim,
        n_layers,
        n_sublayers=None,
        feed_forward_hidden=512,
        normalization="batch",
        norm_eps_alpha=1e-05,
        norm_learn_affine=True,
        norm_track_stats=False,
        norm_momentum_beta=0.1,
        lrnorm_k=1.0,
        gnorm_groups=3,
        activation_function="gelu",
        af_param=1.0,
        af_threshold=6.0,
        af_replacement_value=6.0,
        af_num_params=3,
        af_uniform_range=None,
        dropout_rate=0.1,
        agg=None,
        connection_type="skip",
        expansion_rate=4,
        num_experts=4,
        k=2,
        noisy_gating=True,
        norm_config: Optional[NormalizationConfig] = None,
        activation_config: Optional[ActivationConfig] = None,
        **kwargs,
    ):
        """Initialize the MoEGraphAttentionEncoder."""
        # Store MoE-specific parameters for _create_layer()
        self.num_experts = num_experts
        self.k = k
        self.noisy_gating = noisy_gating
        self.activation_function = activation_function
        self.af_param = af_param
        self.af_threshold = af_threshold
        self.af_replacement_value = af_replacement_value
        self.af_num_params = af_num_params
        self.af_uniform_range = af_uniform_range if af_uniform_range is not None else [0.125, 1 / 3]

        # Store parameters for layer creation
        self.normalization = normalization
        self.norm_eps_alpha = norm_eps_alpha
        self.norm_learn_affine = norm_learn_affine
        self.norm_track_stats = norm_track_stats
        self.norm_momentum_beta = norm_momentum_beta
        self.lrnorm_k = lrnorm_k
        self.gnorm_groups = gnorm_groups

        if norm_config is None:
            norm_config = NormalizationConfig(
                norm_type=normalization,
                epsilon=norm_eps_alpha,
                learn_affine=norm_learn_affine,
                track_stats=norm_track_stats,
                momentum=norm_momentum_beta,
                k_lrnorm=lrnorm_k,
                n_groups=gnorm_groups,
            )

        if activation_config is None:
            activation_config = ActivationConfig(
                name=activation_function,
                param=af_param,
                threshold=af_threshold,
                replacement_value=af_replacement_value,
                n_params=af_num_params,
                range=self.af_uniform_range,
            )

        # Initialize base class (handles layer creation, dropout, forward pass)
        super(MoEGraphAttentionEncoder, self).__init__(
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
        Create a single MoE multi-head attention layer.

        Parameters
        ----------
        layer_idx : int
            Index of the layer being created (0 to n_layers-1).

        Returns
        -------
        nn.Module
            MoEMultiHeadAttentionLayer instance with MoE feed-forward.
        """
        return MoEMultiHeadAttentionLayer(
            self.n_heads,
            self.embed_dim,
            self.feed_forward_hidden,
            self.normalization,
            self.norm_eps_alpha,
            self.norm_learn_affine,
            self.norm_track_stats,
            self.norm_momentum_beta,
            self.lrnorm_k,
            self.gnorm_groups,
            self.activation_function,
            self.af_param,
            self.af_threshold,
            self.af_replacement_value,
            self.af_num_params,
            self.af_uniform_range,
            connection_type=self.conn_type,
            expansion_rate=self.expansion_rate,
            num_experts=self.num_experts,
            k=self.k,
            noisy_gating=self.noisy_gating,
        )
