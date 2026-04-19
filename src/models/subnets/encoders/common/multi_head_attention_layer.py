"""Base Multi-Head Attention Layer for Encoder Architectures."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from logic.src.configs.models.activation_function import ActivationConfig
from logic.src.configs.models.normalization import NormalizationConfig
from logic.src.models.subnets.modules import MultiHeadAttention, Normalization
from logic.src.models.subnets.modules.connections import get_connection_module

from .feed_forward_sublayer import EncoderFeedForwardSubLayer


class MultiHeadAttentionLayerBase(nn.Module):
    """
    Base Multi-Head Attention Layer for Encoder Architectures.

    Implements the standard transformer encoder layer pattern:
    1. Multi-Head Attention + Residual Connection
    2. Layer Normalization
    3. Feed-Forward Network + Residual Connection
    4. Layer Normalization

    This base class is used by multiple encoder types (GAT, MoE) and supports:
    - Configurable normalization (batch, layer, instance, group)
    - Configurable activation functions
    - Skip, dense, or hyper-connections via the connection factory
    - 4D tensor handling for hyper-connections

    Subclasses can override the feed-forward component to use specialized
    variants (e.g., MoEFeedForward, ConvolutionalFFN).

    Parameters
    ----------
    n_heads : int
        Number of attention heads.
    embed_dim : int
        Embedding dimension (must be divisible by n_heads).
    feed_forward_hidden : int
        Hidden dimension for feed-forward network.
    norm_config : Optional[NormalizationConfig], default=None
        Normalization configuration. If None, uses default (batch norm).
    activation_config : Optional[ActivationConfig], default=None
        Activation function configuration. If None, uses default (ReLU).
    connection_type : str, default="skip"
        Type of connection: "skip" (residual), "dense", or "hyper".
    expansion_rate : int, default=4
        Expansion factor for hyper-connections (only used when connection_type="hyper").
    **kwargs
        Additional keyword arguments (for future extensibility).

    Attributes
    ----------
    att : nn.Module
        Multi-head attention module wrapped with connection.
    norm1 : Normalization
        First normalization layer (after attention).
    ff : nn.Module
        Feed-forward module wrapped with connection.
    norm2 : Normalization
        Second normalization layer (after feed-forward).

    Examples
    --------
    >>> from logic.src.configs.models import NormalizationConfig, ActivationConfig
    >>> layer = MultiHeadAttentionLayerBase(
    ...     n_heads=8,
    ...     embed_dim=128,
    ...     feed_forward_hidden=512,
    ...     norm_config=NormalizationConfig(norm_type='layer'),
    ...     activation_config=ActivationConfig(name='gelu')
    ... )
    >>> output = layer(input_tensor, mask=attention_mask)
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
        """Initialize the MultiHeadAttentionLayerBase."""
        super(MultiHeadAttentionLayerBase, self).__init__()

        # Use default configs if none provided
        if norm_config is None:
            norm_config = NormalizationConfig()

        if activation_config is None:
            activation_config = ActivationConfig()

        # Store configs for subclass access
        self.norm_config = norm_config
        self.activation_config = activation_config

        # 1. Multi-Head Attention with connection wrapper
        self.att = get_connection_module(
            module=MultiHeadAttention(
                n_heads=n_heads,
                input_dim=embed_dim,
                embed_dim=embed_dim,
            ),
            embed_dim=embed_dim,
            connection_type=connection_type,
            expansion_rate=expansion_rate,
        )

        # 2. First normalization (after attention)
        self.norm1 = Normalization(
            embed_dim,
            norm_config=norm_config,
        )

        # 3. Feed-Forward Network with connection wrapper
        self.ff = self._create_feed_forward(
            embed_dim=embed_dim,
            feed_forward_hidden=feed_forward_hidden,
            activation_config=activation_config,
            connection_type=connection_type,
            expansion_rate=expansion_rate,
        )

        # 4. Second normalization (after feed-forward)
        self.norm2 = Normalization(
            embed_dim,
            norm_config=norm_config,
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
        Create the feed-forward component.

        Subclasses can override this to use specialized feed-forward variants
        (e.g., MoEFeedForward, ConvolutionalFFN).

        Parameters
        ----------
        embed_dim : int
            Embedding dimension.
        feed_forward_hidden : int
            Hidden dimension for feed-forward network.
        activation_config : ActivationConfig
            Activation function configuration.
        connection_type : str
            Type of connection wrapper.
        expansion_rate : int
            Expansion factor for hyper-connections.

        Returns
        -------
        nn.Module
            Feed-forward module wrapped with connection.
        """
        return get_connection_module(
            module=EncoderFeedForwardSubLayer(
                embed_dim,
                feed_forward_hidden,
                activation_config=activation_config,
            ),
            embed_dim=embed_dim,
            connection_type=connection_type,
            expansion_rate=expansion_rate,
        )

    def forward(self, h: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the attention layer.

        Parameters
        ----------
        h : torch.Tensor
            Input tensor of shape (batch, seq_len, embed_dim) or (batch, seq_len, embed_dim, n)
            for hyper-connections.
        mask : Optional[torch.Tensor], default=None
            Attention mask of shape (batch, seq_len) or (batch, seq_len, seq_len).

        Returns
        -------
        torch.Tensor
            Output tensor with the same shape as input.
        """
        # 1. Multi-Head Attention
        h = self.att(h, mask=mask)

        # 2. First normalization (handle 4D for hyper-connections)
        h = self._apply_norm(h, self.norm1)

        # 3. Feed-Forward Network
        h = self.ff(h)

        # 4. Second normalization (handle 4D for hyper-connections)
        h = self._apply_norm(h, self.norm2)

        return h

    def _apply_norm(self, h: torch.Tensor, norm_layer: nn.Module) -> torch.Tensor:
        """
        Apply normalization, handling both 3D and 4D tensors.

        For hyper-connections, the input is 4D: (batch, seq_len, embed_dim, n).
        We need to permute to (batch, seq_len, n, embed_dim) for normalization,
        then permute back.

        Parameters
        ----------
        h : torch.Tensor
            Input tensor (3D or 4D).
        norm_layer : nn.Module
            Normalization layer to apply.

        Returns
        -------
        torch.Tensor
            Normalized tensor with the same shape as input.
        """
        if h.dim() == 4:  # Hyper-connection: (batch, seq_len, embed_dim, n)
            # Permute to (batch, seq_len, n, embed_dim) for Norm(embed_dim)
            h_perm = h.permute(0, 1, 3, 2).contiguous()
            h_norm: torch.Tensor = norm_layer(h_perm)
            # Permute back to (batch, seq_len, embed_dim, n)
            return h_norm.permute(0, 1, 3, 2)
        else:  # Standard: (batch, seq_len, embed_dim)
            return norm_layer(h)
