"""
SymNCO Policy with projection head.
"""

from typing import Optional

import torch
from tensordict import TensorDict
from torch import nn

from logic.src.models.core.attention_model.policy import AttentionModelPolicy


class SymNCOPolicy(AttentionModelPolicy):
    """
    SymNCO Policy based on AttentionModelPolicy.
    Adds a projection head to the initial embeddings for invariance loss.
    """

    def __init__(self, embed_dim: int = 128, use_projection_head: bool = True, **kwargs):
        """Initialize SymNCOPolicy."""
        super().__init__(embed_dim=embed_dim, **kwargs)
        self.use_projection_head = use_projection_head

        if self.use_projection_head:
            # RL4CO uses a simple 3-layer MLP as projection head
            self.projection_head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim),
            )

    def forward(
        self,
        td: TensorDict,
        env,
        strategy: str = "sampling",
        num_starts: int = 1,
        actions: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict:
        """
        Forward pass with embedding projection support.
        """
        # Request init_embeds from AttentionModelPolicy
        kwargs["return_init_embeds"] = True
        out = super().forward(td, env, strategy=strategy, num_starts=num_starts, actions=actions, **kwargs)

        if self.use_projection_head:
            # [batch, graph_size, embed_dim]
            init_embeds = out["init_embeds"]
            # Project embeddings
            out["proj_embeddings"] = self.projection_head(init_embeds)

        return out
