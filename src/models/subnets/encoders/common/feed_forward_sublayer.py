"""Feed-Forward Sub-Layer for Encoder Architectures."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from logic.src.configs.models.activation_function import ActivationConfig
from logic.src.models.subnets.modules import ActivationFunction, FeedForward


class EncoderFeedForwardSubLayer(nn.Module):
    """
    Reusable Feed-Forward Sub-Layer for Encoder Architectures.

    Implements the standard transformer feed-forward pattern:
    1. Linear expansion: embed_dim → feed_forward_hidden
    2. Non-linear activation function
    3. Linear projection: feed_forward_hidden → embed_dim

    This component is used across multiple encoder types (GAT, GGAC, MoE, etc.)
    to provide consistent feed-forward transformation with configurable activation.

    Parameters
    ----------
    embed_dim : int
        Embedding dimension (input and output size).
    feed_forward_hidden : int
        Hidden dimension for the feed-forward network.
        If 0, creates an identity mapping (embed_dim → embed_dim).
    activation_config : Optional[ActivationConfig], default=None
        Configuration for activation function. If None, uses default (ReLU).
    bias : bool, default=True
        Whether to include bias terms in linear layers.
    **kwargs
        Additional keyword arguments (for future extensibility).

    Attributes
    ----------
    sub_layers : nn.Sequential or nn.Module
        The feed-forward network layers.

    Examples
    --------
    >>> from logic.src.configs.models.activation_function import ActivationConfig
    >>> ff_layer = EncoderFeedForwardSubLayer(
    ...     embed_dim=128,
    ...     feed_forward_hidden=512,
    ...     activation_config=ActivationConfig(name='gelu')
    ... )
    >>> output = ff_layer(input_tensor)  # (batch, seq, 128) → (batch, seq, 128)
    """

    def __init__(
        self,
        embed_dim: int,
        feed_forward_hidden: int,
        activation_config: Optional[ActivationConfig] = None,
        bias: bool = True,
        **kwargs,
    ) -> None:
        """Initialize the EncoderFeedForwardSubLayer."""
        super(EncoderFeedForwardSubLayer, self).__init__()

        # Use default activation config if none provided
        if activation_config is None:
            activation_config = ActivationConfig()

        # Build feed-forward network
        if feed_forward_hidden > 0:
            # Standard FFN: expand → activate → project
            self.sub_layers = nn.Sequential(
                FeedForward(embed_dim, feed_forward_hidden, bias=bias),
                ActivationFunction(
                    activation_config=activation_config,
                ),
                FeedForward(feed_forward_hidden, embed_dim, bias=bias),
            )
        else:
            # Identity mapping when feed_forward_hidden == 0
            # Wrap in Sequential to maintain consistent type for self.sub_layers
            self.sub_layers = nn.Sequential(FeedForward(embed_dim, embed_dim, bias=bias))

    def forward(self, h: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the feed-forward network.

        Parameters
        ----------
        h : torch.Tensor
            Input tensor of shape (batch, seq_len, embed_dim) or (batch, seq_len, embed_dim, n)
            for hyper-connections.
        mask : Optional[torch.Tensor], default=None
            Optional mask tensor (not used in standard FFN, but kept for API compatibility
            with graph convolution variants).

        Returns
        -------
        torch.Tensor
            Output tensor with the same shape as input.
        """
        return self.sub_layers(h)
