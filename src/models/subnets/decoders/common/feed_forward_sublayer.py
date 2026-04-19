"""
Base decoder classes for reducing boilerplate across decoder implementations.

This module provides reusable components for decoder architectures:
- FeedForwardSubLayer: Standard FFN sublayer used across multiple decoders
- Future: DecoderBase class for common decoder patterns

The FeedForwardSubLayer follows the standard transformer pattern:
    Input -> FFN (up-projection) -> Activation -> FFN (down-projection) -> Output
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from logic.src.configs.models.activation_function import ActivationConfig
from logic.src.models.subnets.modules import ActivationFunction, FeedForward


class FeedForwardSubLayer(nn.Module):
    """
    Reusable feed-forward sub-layer for decoder architectures.

    Implements the standard transformer feed-forward pattern:
        FFN(x) = Activation(W1 * x + b1) * W2 + b2

    Where:
    - W1: (embed_dim, feed_forward_hidden)  # Up-projection
    - W2: (feed_forward_hidden, embed_dim)  # Down-projection

    This layer is used in GAT decoders, Deep decoders, and other attention-based
    decoder architectures.

    Parameters
    ----------
    embed_dim : int
        Input and output embedding dimension.
    feed_forward_hidden : int
        Hidden dimension for the feed-forward network.
        If 0 or negative, uses a single linear layer (embed_dim -> embed_dim).
    activation_config : Optional[ActivationConfig], default=None
        Activation function configuration (name, parameters, thresholds).
        Defaults to ReLU if None.
    bias : bool, default=True
        Whether to use bias in linear layers.
    **kwargs
        Additional keyword arguments (for future extensibility).

    Attributes
    ----------
    sub_layers : nn.Sequential
        Sequential container with FFN layers and activation.

    Examples
    --------
    Standard usage:
    >>> config = ActivationConfig(name="gelu")
    >>> ff_layer = FeedForwardSubLayer(
    ...     embed_dim=128,
    ...     feed_forward_hidden=512,
    ...     activation_config=config
    ... )
    >>> output = ff_layer(input_tensor)  # (B, N, 128) -> (B, N, 128)

    No hidden layer (direct mapping):
    >>> ff_layer = FeedForwardSubLayer(embed_dim=128, feed_forward_hidden=0)
    >>> output = ff_layer(input_tensor)  # Single linear layer

    Notes
    -----
    - The mask parameter in forward() is ignored (for API compatibility)
    - For feed_forward_hidden <= 0, creates a single linear layer
    - Activation is only applied when feed_forward_hidden > 0
    """

    def __init__(
        self,
        embed_dim: int,
        feed_forward_hidden: int,
        activation_config: Optional[ActivationConfig] = None,
        bias: bool = True,
        **kwargs,
    ):
        """Initialize the FeedForwardSubLayer."""
        super(FeedForwardSubLayer, self).__init__()

        # Default config if not provided
        if activation_config is None:
            activation_config = ActivationConfig()

        # Standard FFN: Linear -> Activation -> Linear
        if feed_forward_hidden > 0:
            self.sub_layers = nn.Sequential(
                # Up-projection: embed_dim -> feed_forward_hidden
                FeedForward(embed_dim, feed_forward_hidden, bias=bias),
                # Activation function
                ActivationFunction(
                    activation_config.name,
                    activation_config.param,
                    activation_config.threshold,
                    activation_config.replacement_value,
                    activation_config.n_params,
                    tuple(activation_config.range) if activation_config.range else None,  # type: ignore[arg-type]
                ),
                # Down-projection: feed_forward_hidden -> embed_dim
                FeedForward(feed_forward_hidden, embed_dim, bias=bias),
            )
        else:
            # Degenerate case: single linear layer (no activation)
            self.sub_layers = FeedForward(embed_dim, embed_dim, bias=bias)  # type: ignore[assignment]

    def forward(
        self,
        h: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through feed-forward sublayer.

        Parameters
        ----------
        h : torch.Tensor
            Input tensor of shape (batch_size, seq_len, embed_dim).
        mask : Optional[torch.Tensor], default=None
            Mask tensor (currently unused, kept for API compatibility).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, seq_len, embed_dim).

        Notes
        -----
        The mask parameter is ignored in the current implementation.
        It's kept for compatibility with layer interfaces that expect it.
        """
        return self.sub_layers(h)
