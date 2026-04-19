"""
Pointer Attention Mixture of Experts Module.
"""

import math
from typing import Optional

import torch
from torch import nn

from logic.src.models.subnets.modules.moe_layer import MoE
from logic.src.models.subnets.modules.multi_head_attention import MultiHeadAttention


class PointerAttnMoE(nn.Module):
    """
    Pointer Attention with Mixture-of-Experts (MoE) output projection.

    Uses standard Multi-Head Attention to compute a glimpse (context vector),
    then projects this glimpse using an MoE layer before computing the final
    pointing logits. This allows the model to specialize its pointing strategy
    based on the context.

    Args:
        embed_dim: Embedding dimension.
        num_heads: Number of attention heads.
        num_experts: Number of experts in the MoE layer.
        k: Number of experts to select per token.
        noisy_gating: Whether to use noisy gating for MoE.
        mask_inner: Whether to mask the inner MHA.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_experts: int = 4,
        k: int = 2,
        noisy_gating: bool = True,
        mask_inner: bool = True,
        check_nan: bool = True,
    ):
        """
        Initialize PointerAttnMoE.

        Args:
            embed_dim: Embedding dimension.
            num_heads: Number of heads.
            num_experts: Number of experts.
            k: Top-k experts.
            noisy_gating: Noisy gating.
            mask_inner: Mask inner attention.
            check_nan: Check for NaNs.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.check_nan = check_nan
        self.mask_inner = mask_inner

        # Inner Multi-Head Attention to compute glimpse
        # We reuse the existing MultiHeadAttention module
        self.mha = MultiHeadAttention(n_heads=num_heads, input_dim=embed_dim, embed_dim=embed_dim)

        # MoE for output projection of the glimpse
        # This replaces the standard linear projection
        self.moe_out = MoE(
            input_size=embed_dim,
            output_size=embed_dim,
            num_experts=num_experts,
            k=k,
            noisy_gating=noisy_gating,
            out_bias=False,  # Usually no bias for projection before dot product
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        logit_key: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query: Query tensor (batch, num_steps, embed_dim)
            key: Key tensor (batch, graph_size, embed_dim)
            value: Value tensor (batch, graph_size, embed_dim)
            logit_key: Logit key tensor (batch, graph_size, embed_dim) - used for final dot product
            attn_mask: Mask for attention (batch, graph_size) or (batch, num_steps, graph_size)

        Returns:
            logits: (batch, num_steps, graph_size)
        """

        # 1. Compute glimpse using MHA

        # Adjust mask for MHA if needed to handle (batch, graph_size)
        mha_mask = attn_mask
        if self.mask_inner and attn_mask is not None and attn_mask.dim() == 2:
            # (batch, graph_size) -> (batch, num_steps, graph_size)
            num_steps = query.size(1)
            mha_mask = attn_mask.unsqueeze(1).expand(-1, num_steps, -1)

        glimpse = self.mha(q=query, h=key, mask=mha_mask)

        # 2. Project glimpse using MoE
        # MoE forward: (x, loss_coef) -> (y, loss) or just y depending on implementation
        # The MoE in logic.src.models.subnets.modules/moe.py returns just y.reshape(...)

        glimpse_moe = self.moe_out(glimpse)

        # 3. Compute logits via dot product
        # glimpse_moe: (batch, num_steps, embed_dim)
        # logit_key: (batch, graph_size, embed_dim)

        if logit_key.dim() == 3:
            # Standard case
            # (B, N_steps, E) x (B, E, N_nodes) -> (B, N_steps, N_nodes)
            logits = torch.matmul(glimpse_moe, logit_key.transpose(-2, -1))
        else:
            # If logit_key is somehow different shape, handle or fail
            logits = torch.matmul(glimpse_moe, logit_key.transpose(-2, -1))

        # Scale
        logits = logits / math.sqrt(self.embed_dim)

        if self.check_nan:
            assert not torch.isnan(logits).any(), "PointerAttnMoE logits contain NaNs"

        return logits
