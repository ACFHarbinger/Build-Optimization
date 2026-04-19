"""
Mixture of Experts (MoE) Policy.

Extends AttentionModelPolicy with MoE Graph Attention Encoder.
"""

from typing import Any, cast

from logic.src.models.core.attention_model import AttentionModelPolicy
from logic.src.models.subnets.encoders.moe.encoder import MoEGraphAttentionEncoder


class MoEPolicy(AttentionModelPolicy):
    """
    Mixture of Experts Policy based on AttentionModelPolicy.
    Uses MoEGraphAttentionEncoder instead of standard GraphAttentionEncoder.
    """

    def __init__(
        self,
        env_name: str,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        n_encode_layers: int = 3,
        n_heads: int = 8,
        normalization: str = "batch",
        num_experts: int = 4,
        k: int = 2,
        noisy_gating: bool = True,
        **kwargs,
    ):
        """Initialize MoEPolicy."""
        # We invoke super() but we will overwrite the encoder.
        # This is slightly inefficient (creates standard encoder then throws it away)
        # but prevents duplicating all the other setup logic in AttentionModelPolicy.
        super().__init__(
            env_name=env_name,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            n_encode_layers=n_encode_layers,
            n_heads=n_heads,
            normalization=normalization,
            **kwargs,
        )

        # Overwrite encoder with MoE version
        self.encoder = cast(
            Any,
            MoEGraphAttentionEncoder(
                n_heads=n_heads,
                embed_dim=embed_dim,
                feed_forward_hidden=hidden_dim,
                n_layers=n_encode_layers,
                normalization=normalization,
                num_experts=num_experts,
                k=k,
                noisy_gating=noisy_gating,
                **kwargs,
            ),
        )
