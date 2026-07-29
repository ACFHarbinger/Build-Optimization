"""
MoE Feed Forward Network implementation.
"""

from torch import nn

from .activation_function import ActivationFunction
from .feed_forward import FeedForward
from .moe_layer import MoE


class MoEFeedForward(nn.Module):
    """
    Sub-layer replacing standard FeedForward with Mixture of Experts.
    """

    def __init__(
        self,
        embed_dim,
        feed_forward_hidden,
        activation,
        af_param,
        threshold,
        replacement_value,
        n_params,
        dist_range,
        bias=True,
        num_experts=4,
        k=2,
        noisy_gating=True,
    ):
        """Initializes the MoE FeedForwardSubLayer."""
        super(MoEFeedForward, self).__init__()

        # Create a list of experts.
        # Each expert is a standard FFN block: Linear -> Act -> Linear
        # Note: In standard Transformer FFN, it expands to feed_forward_hidden and then projects back to embed_dim.
        # Here we make each expert a full FFN.

        experts = nn.ModuleList(
            [
                nn.Sequential(
                    FeedForward(embed_dim, feed_forward_hidden, bias=bias),
                    ActivationFunction(
                        activation,
                        af_param,
                        threshold,
                        replacement_value,
                        n_params,
                        dist_range,
                    ),
                    FeedForward(feed_forward_hidden, embed_dim, bias=bias),
                )
                for _ in range(num_experts)
            ]
        )

        # Initialize MoE with these experts
        # input_size = embed_dim, output_size = embed_dim
        self.moe = MoE(
            input_size=embed_dim,
            output_size=embed_dim,
            experts=experts,
            num_experts=num_experts,
            k=k,
            noisy_gating=noisy_gating,
        )

    def forward(self, h, mask=None):
        """Forward pass."""
        # MoE logic handles the dispatching and combination
        return self.moe(h)
