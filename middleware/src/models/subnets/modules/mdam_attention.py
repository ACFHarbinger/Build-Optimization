"""
MDAM Multi-Head Attention module.

This module implements the MDAM-specific MHA that can optionally return
intermediate attention weights and value matrices for encoder-decoder coupling.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn


class MultiHeadAttentionMDAM(nn.Module):
    """
    Multi-Head Attention for MDAM with optional return of attention weights and values.

    Unlike standard MHA, this module can return intermediate attention weights (attn)
    and value matrices (V) when `last_one=True`. These are used by the MDAM encoder's
    `change()` method to dynamically update embeddings during decoding.

    Reference:
        Xin et al. "Multi-Decoder Attention Model with Embedding Glimpse for
        Solving Vehicle Routing Problems" (AAAI 2021)
    """

    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        last_one: bool = False,
    ) -> None:
        """
        Initialize MDAM Multi-Head Attention.

        Args:
            embed_dim: Total embedding dimension.
            n_heads: Number of attention heads.
            last_one: If True, forward returns (out, attn, V) instead of just out.
                     Used for the last encoder layer to enable dynamic re-encoding.
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.num_heads = n_heads  # Alias for compatibility
        self.val_dim = embed_dim  # Full embed_dim per head for MDAM
        self.norm_factor = 1 / math.sqrt(embed_dim)

        # Separate weight matrices for each head
        self.W_query = nn.Parameter(torch.Tensor(n_heads, embed_dim, embed_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, embed_dim, embed_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, embed_dim, embed_dim))
        self.W_out = nn.Parameter(torch.Tensor(n_heads, embed_dim, embed_dim))

        self._init_parameters()
        self.last_one = last_one

    def _init_parameters(self) -> None:
        """Initialize parameters with uniform distribution."""
        for param in self.parameters():
            stdv = 1.0 / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(
        self,
        q: torch.Tensor,
        h: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Compute multi-head attention.

        Args:
            q: Query tensor of shape (batch_size, n_query, embed_dim).
            h: Key/Value tensor of shape (batch_size, graph_size, embed_dim).
               If None, computes self-attention (h = q).
            mask: Attention mask of shape (batch_size, n_query, graph_size).
                  True values indicate positions to mask out.

        Returns:
            If last_one=False:
                Output tensor of shape (batch_size, n_query, embed_dim).
            If last_one=True:
                Tuple of (output, attn, V) where:
                - output: (batch_size, n_query, embed_dim)
                - attn: (n_heads, batch_size, n_query, graph_size)
                - V: (n_heads, batch_size, graph_size, embed_dim)
        """
        if h is None:
            h = q  # Self-attention

        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)

        assert q.size(0) == batch_size
        assert q.size(2) == input_dim
        assert input_dim == self.embed_dim, "Wrong embedding dimension of input"

        # Flatten for matrix multiplication
        hflat = h.contiguous().view(-1, input_dim)
        qflat = q.contiguous().view(-1, input_dim)

        # Shapes for reshaping after matmul
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)

        # Compute Q, K, V projections
        # Shape: (n_heads, batch_size, seq_len, embed_dim)
        Q = torch.matmul(qflat, self.W_query).view(shp_q)
        K = torch.matmul(hflat, self.W_key).view(shp)
        V = torch.matmul(hflat, self.W_val).view(shp)

        # Compute attention scores
        # Shape: (n_heads, batch_size, n_query, graph_size)
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))

        # Apply mask if provided
        if mask is not None:
            mask_expanded = mask.view(1, batch_size, n_query, graph_size).expand_as(compatibility)
            compatibility[mask_expanded] = float("-inf")

        # Compute attention weights
        attn = F.softmax(compatibility, dim=-1)

        # Handle NaN from softmax when all values are masked
        if mask is not None:
            attnc = attn.clone()
            mask_expanded = mask.view(1, batch_size, n_query, graph_size).expand_as(compatibility)
            attnc[mask_expanded] = 0
            attn = attnc

        # Compute attention output
        # heads: (n_heads, batch_size, n_query, embed_dim)
        heads = torch.matmul(attn, V)

        # Project and reshape to output
        # (batch_size, n_query, n_heads * embed_dim) -> (batch_size, n_query, embed_dim)
        out = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.embed_dim),
            self.W_out.view(-1, self.embed_dim),
        ).view(batch_size, n_query, self.embed_dim)

        if self.last_one:
            return out, attn, V
        return out
