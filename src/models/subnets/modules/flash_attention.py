"""
Multi-Head Flash Attention module.

Uses PyTorch's scaled_dot_product_attention (SDPA) for efficiency.
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


class MultiHeadFlashAttention(nn.Module):
    """
    Multi-Head Attention using PyTorch's SDPA (Flash Attention).
    """

    last_attn: Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]

    def __init__(
        self,
        n_heads: int,
        input_dim: int,
        embed_dim: int,
        val_dim: Optional[int] = None,
        key_dim: Optional[int] = None,
        bias: bool = True,
        attention_dropout: float = 0.0,
        store_attn_weights: bool = False,
        sdpa_fn: Optional[Callable] = None,
    ) -> None:
        """
        Args:
            n_heads: Number of attention heads.
            input_dim: Input feature dimension.
            embed_dim: Output embedding dimension.
            bias: Use bias in projections.
            attention_dropout: Dropout probability.
            store_attn_weights: If True, falls back to manual attention to store weights.
        """
        super().__init__()

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // n_heads
        self.store_attn_weights = store_attn_weights
        self.attention_dropout = attention_dropout

        assert self.head_dim * n_heads == self.embed_dim, "embed_dim must be divisible by n_heads"

        self.W_query = nn.Linear(input_dim, embed_dim, bias=bias)
        self.W_key = nn.Linear(input_dim, embed_dim, bias=bias)
        self.W_val = nn.Linear(input_dim, embed_dim, bias=bias)
        self.W_out = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.init_parameters()
        self.last_attn = (None, None)

        self.sdpa_fn = sdpa_fn if sdpa_fn is not None else scaled_dot_product_attention
        if self.sdpa_fn is None and not store_attn_weights:
            # Warning or silent fallback? Silent for now.
            pass

    def init_parameters(self) -> None:
        """Initializes the parameters using Xavier uniform initialization."""
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(
        self,
        q: torch.Tensor,
        h: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            q: Queries [batch, n_query, input_dim]
            h: Keys/Values [batch, graph_size, input_dim]. If None, self-attention.
            mask: Attention mask [batch, n_query, graph_size]. True = masked out.
        """
        if h is None:
            h = q

        batch_size = q.size(0)
        n_query = q.size(1)
        graph_size = h.size(1)

        # q: [batch, len, dim] -> [batch, len, heads, head_dim] -> [batch, heads, len, head_dim]
        Q = self.W_query(q).view(batch_size, n_query, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.W_key(h).view(batch_size, graph_size, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.W_val(h).view(batch_size, graph_size, self.n_heads, self.head_dim).transpose(1, 2)

        input_mask = None
        if mask is not None:
            if mask.dim() == 2:
                input_mask = mask.view(batch_size, 1, 1, graph_size)
            elif mask.dim() == 3:
                input_mask = mask.unsqueeze(1)
            else:
                input_mask = mask

        if self.store_attn_weights or self.sdpa_fn is None:
            return self._manual_attention(Q, K, V, input_mask)
        else:
            sdpa_mask = ~input_mask if input_mask is not None else None

            out = self.sdpa_fn(Q, K, V, attn_mask=sdpa_mask, dropout_p=self.attention_dropout if self.training else 0.0)

            out = out.transpose(1, 2).contiguous().view(batch_size, n_query, self.embed_dim)
            return self.W_out(out)

    def _manual_attention(self, Q, K, V, mask):
        """manual attention.

        Args:
            Q (Any): Description of Q.
            K (Any): Description of K.
            V (Any): Description of V.
            mask (Any): Description of mask.

        Returns:
            Any: Description of return value.
        """
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)
        self.last_attn = (attn_weights.detach(), mask.detach() if mask is not None else None)

        if self.attention_dropout > 0.0 and self.training:
            attn_weights = torch.nn.functional.dropout(attn_weights, p=self.attention_dropout)

        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(Q.size(0), Q.size(2), self.embed_dim)
        return self.W_out(out)
