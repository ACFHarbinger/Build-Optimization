"""
Multi-Head Cross Attention module.
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Tuple

import torch
from torch import nn

try:
    from torch.nn.functional import scaled_dot_product_attention
except ImportError:
    scaled_dot_product_attention = None  # type: ignore[assignment]


class MultiHeadCrossAttention(nn.Module):
    """
    Multi-Head Cross Attention with separate query and key/value projections.
    Uses PyTorch's native `scaled_dot_product_attention` (SDPA).
    """

    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        bias: bool = False,
        attention_dropout: float = 0.0,
        store_attn_weights: bool = False,
        sdpa_fn: Optional[Callable] = None,
    ) -> None:
        """
        Args:
            embed_dim: Total embedding dimension.
            n_heads: Number of attention heads.
            bias: Whether to use bias in linear projections.
            attention_dropout: Dropout rate for attention weights.
            store_attn_weights: If True, uses manual attention to store weights (slower).
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.store_attn_weights = store_attn_weights
        self.attention_dropout = attention_dropout

        assert self.embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"
        self.head_dim = self.embed_dim // n_heads

        # Separate projections for Q and KV
        self.Wq = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.Wkv = nn.Linear(embed_dim, 2 * embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.sdpa_fn = sdpa_fn if sdpa_fn is not None else scaled_dot_product_attention
        self.last_attn: Tuple[Optional[torch.Tensor], Optional[torch.Tensor]] = (
            None,
            None,
        )

    def forward(
        self,
        q_input: torch.Tensor,
        kv_input: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            q_input: Query input (batch, n_query, embed_dim)
            kv_input: Key/Value input (batch, graph_size, embed_dim)
            mask: Attention mask (batch, n_query, graph_size).
                  WSmart convention: True = masked (invalid).
        """
        batch_size = q_input.size(0)
        n_query = q_input.size(1)
        graph_size = kv_input.size(1)

        # Project queries
        # [batch, n_query, embed_dim] -> [batch, n_query, n_heads, head_dim] -> [batch, n_heads, n_query, head_dim]
        q = self.Wq(q_input).view(batch_size, n_query, self.n_heads, self.head_dim).transpose(1, 2)

        # Project keys/values
        # [batch, graph_size, 2*embed_dim]
        kv = self.Wkv(kv_input)
        # Split into k and v: [batch, graph_size, embed_dim] each
        k_full, v_full = kv.chunk(2, dim=-1)

        # Reshape to [batch, n_heads, graph_size, head_dim]
        k = k_full.view(batch_size, graph_size, self.n_heads, self.head_dim).transpose(1, 2)
        v = v_full.view(batch_size, graph_size, self.n_heads, self.head_dim).transpose(1, 2)

        # Handle mask
        input_mask = None
        if mask is not None:
            if mask.dim() == 2:
                # [batch, graph_size] -> [batch, 1, 1, graph_size]
                input_mask = mask.view(batch_size, 1, 1, graph_size)
            elif mask.dim() == 3:
                # [batch, n_query, graph_size] -> [batch, 1, n_query, graph_size]
                input_mask = mask.unsqueeze(1)
            else:
                input_mask = mask

        if self.store_attn_weights or self.sdpa_fn is None:
            return self._manual_attention(q, k, v, input_mask)
        else:
            # Invert mask for SDPA if provided
            sdpa_mask = ~input_mask if input_mask is not None else None

            # Using SDPA
            out = self.sdpa_fn(
                q,
                k,
                v,
                attn_mask=sdpa_mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
            )
            # Reshape back: [batch, n_heads, n_query, head_dim] -> [batch, n_query, n_heads, head_dim] -> [batch, n_query, embed_dim]
            out = out.transpose(1, 2).contiguous().view(batch_size, n_query, self.embed_dim)
            return self.out_proj(out)

    def _manual_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Manual attention implementation for weights storage or fallback."""

        # Scale q
        q = q * (1.0 / math.sqrt(self.head_dim))

        # [b, h, m, n]
        attn_logits = torch.matmul(q, k.transpose(-2, -1))

        if mask is not None:
            # mask is [b, 1, m, n] (from input_mask handling above)
            # WSmart convention: True = masked (set to -inf)
            attn_logits = attn_logits.masked_fill(mask, float("-inf"))

        attn_weights = torch.softmax(attn_logits, dim=-1)

        # Store weights
        self.last_attn = (attn_weights.detach().cpu(), mask.detach().cpu() if mask is not None else None)

        if self.attention_dropout > 0.0 and self.training:
            attn_weights = torch.nn.functional.dropout(attn_weights, p=self.attention_dropout)

        out = torch.matmul(attn_weights, v)
        # [b, h, m, d] -> [b, m, h, d] -> [b, m, embed]
        out = out.transpose(1, 2).contiguous().view(q.size(0), q.size(2), self.embed_dim)
        return self.out_proj(out)
