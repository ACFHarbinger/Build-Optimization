"""
PolyNet Attention Module.

Implements K-strategy conditioning via binary vectors for diverse solution learning.
Based on Hottung et al. (2024): https://arxiv.org/abs/2402.14048
"""

from __future__ import annotations

import itertools
import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


class PolyNetAttention(nn.Module):
    """
    PolyNet Attention with K-strategy binary conditioning.

    Extends pointer attention to condition logits on K different binary vectors,
    allowing learning of K diverse solution strategies from a single model.

    The binary vectors are generated as the first K elements of the set of all
    binary vectors of length ceil(log2(K)).

    Reference:
        Hottung et al. "PolyNet: Learning Diverse Solution Strategies for
        Neural Combinatorial Optimization" (2024)
    """

    binary_vectors: torch.Tensor

    def __init__(
        self,
        k: int,
        embed_dim: int,
        poly_layer_dim: int = 256,
        num_heads: int = 8,
        mask_inner: bool = True,
        out_bias: bool = False,
        check_nan: bool = True,
    ) -> None:
        """
        Initialize PolyNet attention.

        Args:
            k: Number of strategies to learn.
            embed_dim: Embedding dimension.
            poly_layer_dim: Hidden dimension of PolyNet layers.
            num_heads: Number of attention heads.
            mask_inner: Whether to mask inner attention.
            out_bias: Whether to use bias in output projection.
            check_nan: Whether to check for NaN values.
        """
        super().__init__()

        self.k = k
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mask_inner = mask_inner
        self.check_nan = check_nan

        # Binary vector dimension: ceil(log2(k)) bits needed
        self.binary_vector_dim = max(1, math.ceil(math.log2(k)))

        # Generate K binary vectors as non-trainable parameter
        all_binary = list(itertools.product([0, 1], repeat=self.binary_vector_dim))
        binary_vectors = torch.tensor(all_binary[:k], dtype=torch.float32)
        self.register_buffer("binary_vectors", binary_vectors)

        # Output projection
        self.project_out = nn.Linear(embed_dim, embed_dim, bias=out_bias)

        # PolyNet layers: condition on strategy via binary vector
        self.poly_layer_1 = nn.Linear(embed_dim + self.binary_vector_dim, poly_layer_dim)
        self.poly_layer_2 = nn.Linear(poly_layer_dim, embed_dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        logit_key: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute attention logits with K-strategy conditioning.

        Args:
            query: Query tensor (batch, num_steps, embed_dim).
            key: Key tensor (batch, graph_size, embed_dim).
            value: Value tensor (batch, graph_size, embed_dim).
            logit_key: Logit key tensor (batch, graph_size, embed_dim).
            attn_mask: Attention mask (batch, graph_size). True = valid.

        Returns:
            Logits tensor (batch, num_steps, graph_size).
        """
        # Compute inner multi-head attention
        heads = self._inner_mha(query, key, value, attn_mask)

        # Project to glimpse
        glimpse = self.project_out(heads)

        # Apply K-strategy conditioning
        num_solutions = glimpse.shape[1]
        z = self._get_strategy_vectors(num_solutions, glimpse.device)
        z = z[None].expand(glimpse.shape[0], num_solutions, self.binary_vector_dim)

        # PolyNet conditioning layers
        poly_input = torch.cat([glimpse, z], dim=-1)
        poly_out = F.relu(self.poly_layer_1(poly_input))
        poly_out = self.poly_layer_2(poly_out)

        # Residual connection
        glimpse = glimpse + poly_out

        # Compute final logits via dot product with logit key
        logits = torch.bmm(glimpse, logit_key.squeeze(-2).transpose(-2, -1)).squeeze(-2) / math.sqrt(glimpse.size(-1))

        if self.check_nan:
            assert not torch.isnan(logits).any(), "PolyNet logits contain NaNs"

        return logits

    def _get_strategy_vectors(
        self,
        num_solutions: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Get binary strategy vectors for given number of solutions."""
        # Repeat binary vectors to cover all solutions
        repeats = math.ceil(num_solutions / self.k)
        vectors = self.binary_vectors.repeat(repeats, 1)[:num_solutions]
        return vectors.to(device)

    def _inner_mha(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Perform inner multi-head attention."""
        q = self._make_heads(query)
        k = self._make_heads(key)
        v = self._make_heads(value)

        # Prepare mask for multi-head attention
        if self.mask_inner and attn_mask is not None:
            # Add dimensions for heads and query positions
            if attn_mask.ndim == 2:
                attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)
            elif attn_mask.ndim == 3:
                attn_mask = attn_mask.unsqueeze(1)
        else:
            attn_mask = None

        # Scaled dot-product attention
        scale = math.sqrt(q.size(-1))
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale

        if attn_mask is not None:
            scores = scores.masked_fill(~attn_mask, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        heads = torch.matmul(attn_weights, v)

        # Reshape back: (batch, heads, seq, dim) -> (batch, seq, heads * dim)
        batch_size = heads.size(0)
        seq_len = heads.size(2)
        heads = heads.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        return heads

    def _make_heads(self, v: torch.Tensor) -> torch.Tensor:
        """Reshape tensor for multi-head attention."""
        batch_size = v.size(0)
        seq_len = v.size(1)
        head_dim = self.embed_dim // self.num_heads
        return v.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)
