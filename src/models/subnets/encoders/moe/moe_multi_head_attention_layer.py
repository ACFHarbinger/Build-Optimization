"""MoE Multi-Head Attention Layer."""

from __future__ import annotations

from torch import nn

from logic.src.configs.models.activation_function import ActivationConfig
from logic.src.configs.models.normalization import NormalizationConfig
from logic.src.models.subnets.encoders.common import MultiHeadAttentionLayerBase
from logic.src.models.subnets.modules.connections import get_connection_module
from logic.src.models.subnets.modules.moe_feed_forward import MoEFeedForward


class MoEMultiHeadAttentionLayer(MultiHeadAttentionLayerBase):
    """
    Single layer of the MoE Graph Attention Encoder.

    Inherits from MultiHeadAttentionLayerBase but overrides the feed-forward
    component to use Mixture-of-Experts (MoE) routing instead of standard FFN.

    The MoE variant allows the model to specialize different experts for different
    input patterns, potentially improving capacity and expressiveness.

    Parameters
    ----------
    n_heads : int
        Number of attention heads.
    embed_dim : int
        Embedding dimension.
    feed_forward_hidden : int
        Hidden dimension for each expert's feed-forward network.
    normalization : str
        Normalization type.
    epsilon_alpha : float
        Epsilon for normalization stability.
    learn_affine : bool
        Whether to learn affine parameters in normalization.
    track_stats : bool
        Whether to track running stats in batch norm.
    mbeta : float
        Momentum for batch norm running stats.
    lr_k : float
        Local response normalization parameter.
    n_groups : int
        Number of groups for group normalization.
    activation : str
        Activation function name.
    af_param : float
        Activation function parameter.
    threshold : float
        Activation threshold.
    replacement_value : float
        Activation replacement value.
    n_params : int
        Number of activation parameters.
    uniform_range : list
        Uniform range for activation.
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
        connection_type="skip",
        expansion_rate=4,
        num_experts=4,
        k=2,
        noisy_gating=True,
    ):
        """Initialize the MoEMultiHeadAttentionLayer."""
        # Store MoE-specific parameters for _create_feed_forward()
        self.num_experts = num_experts
        self.k = k
        self.noisy_gating = noisy_gating
        self.activation = activation
        self.af_param = af_param
        self.threshold = threshold
        self.replacement_value = replacement_value
        self.n_params = n_params
        self.uniform_range = uniform_range

        # Create config objects
        norm_config = NormalizationConfig(
            norm_type=normalization,
            epsilon=epsilon_alpha,
            learn_affine=learn_affine,
            track_stats=track_stats,
            momentum=mbeta,
            k_lrnorm=lr_k,
            n_groups=n_groups,
        )

        activation_config = ActivationConfig(
            name=activation,
            param=af_param,
            threshold=threshold,
            replacement_value=replacement_value,
            n_params=n_params,
            range=uniform_range,
        )

        # Delegate to base class
        super(MoEMultiHeadAttentionLayer, self).__init__(
            n_heads=n_heads,
            embed_dim=embed_dim,
            feed_forward_hidden=feed_forward_hidden,
            norm_config=norm_config,
            activation_config=activation_config,
            connection_type=connection_type,
            expansion_rate=expansion_rate,
        )

    def _create_feed_forward(
        self,
        embed_dim: int,
        feed_forward_hidden: int,
        activation_config: ActivationConfig,
        connection_type: str,
        expansion_rate: int,
    ) -> nn.Module:
        """
        Override to use MoEFeedForward instead of standard feed-forward.

        Parameters
        ----------
        embed_dim : int
            Embedding dimension.
        feed_forward_hidden : int
            Hidden dimension for each expert.
        activation_config : ActivationConfig
            Activation function configuration (already stored in self).
        connection_type : str
            Connection type wrapper.
        expansion_rate : int
            Expansion factor for hyper-connections.

        Returns
        -------
        nn.Module
            MoEFeedForward module wrapped with connection.
        """
        return get_connection_module(
            module=MoEFeedForward(
                embed_dim,
                feed_forward_hidden,
                self.activation,
                self.af_param,
                self.threshold,
                self.replacement_value,
                self.n_params,
                self.uniform_range,
                num_experts=self.num_experts,
                k=self.k,
                noisy_gating=self.noisy_gating,
            ),
            embed_dim=embed_dim,
            connection_type=connection_type,
            expansion_rate=expansion_rate,
        )
