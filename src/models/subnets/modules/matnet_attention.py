"""
MatNet Attention Module.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn


class MixedScoreMHA(nn.Module):
    """
    Mixed-Score Multi-Head Attention for MatNet.
    Computes attention scores by combining row and column representations with matrix elements.
    Reference: MatNet (Kwon et al., 2021)
    """

    def __init__(
        self,
        n_heads: int,
        embed_dim: int,
        key_dim: Optional[int] = None,
    ):
        """
        Initialize MixedScoreMHA.

        Args:
            n_heads: Number of heads.
            embed_dim: Embedding dimension.
            key_dim: Key dimension.
        """
        super().__init__()
        if key_dim is None:
            key_dim = embed_dim // n_heads

        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.key_dim = key_dim
        self.norm_factor = 1 / math.sqrt(key_dim)

        # Projections for row embeddings
        self.W_query_row = nn.Linear(embed_dim, n_heads * key_dim, bias=False)
        self.W_key_row = nn.Linear(embed_dim, n_heads * key_dim, bias=False)
        self.W_val_row = nn.Linear(embed_dim, n_heads * key_dim, bias=False)

        # Projections for column embeddings
        self.W_query_col = nn.Linear(embed_dim, n_heads * key_dim, bias=False)
        self.W_key_col = nn.Linear(embed_dim, n_heads * key_dim, bias=False)
        self.W_val_col = nn.Linear(embed_dim, n_heads * key_dim, bias=False)

        # Weight for the distance matrix scores
        self.W_mat = nn.Parameter(torch.Tensor(n_heads, 1, 1))

        self.W_out_row = nn.Linear(n_heads * key_dim, embed_dim, bias=False)
        self.W_out_col = nn.Linear(n_heads * key_dim, embed_dim, bias=False)

        self.init_parameters()

    def init_parameters(self):
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.W_mat)

    def forward(
        self, row_emb: torch.Tensor, col_emb: torch.Tensor, matrix: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            row_emb: [batch, row_size, embed_dim]
            col_emb: [batch, col_size, embed_dim]
            matrix: [batch, row_size, col_size]
        """
        batch_size, row_size, _ = row_emb.size()

        # Projections and reshape to [batch, n_heads, size, key_dim]
        def project(linear, emb):
            """Project and reshape."""
            out = linear(emb)
            return out.view(batch_size, -1, self.n_heads, self.key_dim).transpose(1, 2)

        row_q = project(self.W_query_row, row_emb)
        row_k = project(self.W_key_row, row_emb)
        row_v = project(self.W_val_row, row_emb)

        col_q = project(self.W_query_col, col_emb)
        col_k = project(self.W_key_col, col_emb)
        col_v = project(self.W_val_col, col_emb)

        # Mixed scores: (Row_Q @ Col_K^T) + (Col_Q @ Row_K^T)^T + Matrix_contribution
        score1 = torch.matmul(row_q, col_k.transpose(-2, -1))  # [batch, n_heads, row_size, col_size]
        score2 = torch.matmul(col_q, row_k.transpose(-2, -1)).transpose(-2, -1)  # [batch, n_heads, row_size, col_size]
        score3 = matrix.unsqueeze(1) * self.W_mat  # [batch, n_heads, row_size, col_size]

        compatibility = self.norm_factor * (score1 + score2 + score3)

        if mask is not None:
            # mask is likely [batch, row_size, col_size] or similar
            if mask.dim() == 3:
                compatibility.masked_fill_(mask.unsqueeze(1), -1e9)
            else:
                compatibility.masked_fill_(mask, -1e9)

        # Attention row-wise (row attending to cols)
        attn_row = torch.softmax(compatibility, dim=-1)
        # Attention col-wise (col attending to rows)
        attn_col = torch.softmax(compatibility, dim=-2)

        # New row embeddings from col values
        row_heads = torch.matmul(attn_row, col_v)  # [batch, n_heads, row_size, key_dim]
        # New col embeddings from row values
        col_heads = torch.matmul(attn_col.transpose(-2, -1), row_v)  # [batch, n_heads, col_size, key_dim]

        # Reshape and project out
        def project_out(linear, heads):
            """Project out."""
            # heads is [batch, n_heads, size, key_dim]
            out = heads.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.key_dim)
            return linear(out)

        row_out = project_out(self.W_out_row, row_heads)
        col_out = project_out(self.W_out_col, col_heads)

        return row_out, col_out
