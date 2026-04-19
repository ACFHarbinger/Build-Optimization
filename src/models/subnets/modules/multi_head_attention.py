"""
Multi-Head Attention module for transformer-based models.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism.

    Allows the model to jointly attend to information from different representation subspaces
    at different positions. Implements scaled dot-product attention.
    """

    last_attn: Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]

    def __init__(
        self,
        n_heads: int,
        input_dim: int,
        embed_dim: int,
        val_dim: Optional[int] = None,
        key_dim: Optional[int] = None,
    ) -> None:
        """
        Args:
            n_heads: Number of attention heads.
            input_dim: Dimensionality of the input features.
            embed_dim: Total embedding dimension (output dimension).
            val_dim: Dimension of value vectors per head.
            key_dim: Dimension of key/query vectors per head.
        """
        super(MultiHeadAttention, self).__init__()
        if val_dim is None:
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim
        self.norm_factor: float = 1 / math.sqrt(key_dim)

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))
        self.W_out = nn.Parameter(torch.Tensor(n_heads, key_dim, embed_dim))
        self.init_parameters()
        self.last_attn = (None, None)

    def init_parameters(self) -> None:
        """Initializes the parameters of the attention layers using Xavier uniform initialization."""
        for param in self.parameters():
            if param.dim() > 0:
                stdv: float = 1.0 / math.sqrt(param.size(-1))
                param.data.uniform_(-stdv, stdv)

    def forward(
        self,
        q: torch.Tensor,
        h: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Computes the multi-head attention.

        Args:
            q: Queries tensor of shape (batch_size, n_query, input_dim).
            h: Key/Value keys tensor of shape (batch_size, graph_size, input_dim).
               If None, defaults to q (self-attention).
            mask: Attention mask of shape (batch_size, n_query, graph_size).

        Returns:
            context vectors (aggregated values) of shape (batch_size, n_query, embed_dim).
        """
        if h is None:
            h = q

        batch_size, graph_size, input_dim = h.size()
        n_query: int = q.size(1)
        assert q.size(0) == batch_size
        assert q.size(2) == input_dim
        assert input_dim == self.input_dim, "Wrong embedding dimension of input"

        hflat: torch.Tensor = h.contiguous().view(-1, input_dim)
        qflat: torch.Tensor = q.contiguous().view(-1, input_dim)

        shp: Tuple[int, int, int, int] = (self.n_heads, batch_size, graph_size, -1)
        shp_q: Tuple[int, int, int, int] = (self.n_heads, batch_size, n_query, -1)

        Q: torch.Tensor = torch.matmul(qflat, self.W_query).view(shp_q)
        K: torch.Tensor = torch.matmul(hflat, self.W_key).view(shp)
        V: torch.Tensor = torch.matmul(hflat, self.W_val).view(shp)

        compatibility: torch.Tensor = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))

        if mask is not None:
            if mask.dim() == 2:
                mask = mask.view(1, batch_size, n_query, graph_size)

            mask_expanded: torch.Tensor = mask.expand_as(compatibility)
            compatibility[mask_expanded] = -math.inf

        attn: torch.Tensor = torch.softmax(compatibility, dim=-1)

        if mask is not None:
            mask_expanded = mask.expand_as(compatibility)
            self.last_attn = (attn.detach().clone(), mask_expanded.detach().clone())
            attnc: torch.Tensor = attn.clone()
            attnc[mask_expanded] = 0
            attn = attnc
        else:
            self.last_attn = (attn.detach().clone(), None)

        heads: torch.Tensor = torch.matmul(attn[:, :, :, :graph_size], V)
        out: torch.Tensor = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim),
        ).view(batch_size, n_query, self.embed_dim)

        return out
