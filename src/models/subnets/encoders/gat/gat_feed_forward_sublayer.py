"""Feed-Forward Sub-Layer for GAT Encoder."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from logic.src.configs.models.activation_function import ActivationConfig
from logic.src.models.subnets.modules import ActivationFunction, FeedForward


class GATFeedForwardSubLayer(nn.Module):
    """
    Feed-Forward Sub-Layer for GAT Encoder.
    Contains:
    - FeedForward (expansion)
    - Activation
    - FeedForward (projection)
    """

    def __init__(
        self,
        embed_dim: int,
        feed_forward_hidden: int,
        activation_config: Optional[ActivationConfig] = None,
        bias: bool = True,
        **kwargs,
    ) -> None:
        """Initializes the GATFeedForwardSubLayer."""
        super(GATFeedForwardSubLayer, self).__init__()

        if activation_config is None:
            activation_config = ActivationConfig()

        self.sub_layers = (
            nn.Sequential(
                FeedForward(embed_dim, feed_forward_hidden, bias=bias),
                ActivationFunction(
                    activation_config.name,
                    activation_config.param,
                    activation_config.threshold,
                    activation_config.replacement_value,
                    activation_config.n_params,
                    (activation_config.range[0], activation_config.range[1])
                    if activation_config.range and len(activation_config.range) >= 2
                    else None,
                ),
                FeedForward(feed_forward_hidden, embed_dim, bias=bias),
            )
            if feed_forward_hidden > 0
            else FeedForward(embed_dim, embed_dim)
        )

    def forward(self, h: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass."""
        return self.sub_layers(h)
