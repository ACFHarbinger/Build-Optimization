"""
MatNet Embedding.
"""

import torch
from torch import nn


class MatNetInitEmbedding(nn.Module):
    """
    Initial embedding layer for MatNet.
    Projects row and column statistics (mean, min, etc.) of the cost matrix.
    Reference: MatNet: Matrix Encoding Network for Combinatorial Optimization (Kwon et al., 2021)
    """

    def __init__(self, embed_dim: int, normalization: str = "instance"):
        """
        Initialize MatNetInitEmbedding.

        Args:
            embed_dim: Embedding dimension.
            normalization: Normalization type.
        """
        super().__init__()
        # We project: mean, min, (and potentially others like max or std)
        # Standard MatNet paper uses mean and min.
        self.row_stats_proj = nn.Linear(2, embed_dim)
        self.col_stats_proj = nn.Linear(2, embed_dim)

    def forward(self, matrix):
        """
        Args:
            matrix: [batch, row_size, col_size]
        Returns:
            row_emb: [batch, row_size, embed_dim]
            col_emb: [batch, col_size, embed_dim]
        """
        # Row stats: mean and min
        row_mean = matrix.mean(dim=2, keepdim=True)
        row_min, _ = matrix.min(dim=2, keepdim=True)
        row_stats = torch.cat([row_mean, row_min], dim=2)  # [batch, row_size, 2]
        row_emb = self.row_stats_proj(row_stats)

        # Col stats: mean and min
        col_mean = matrix.mean(dim=1, keepdim=True).transpose(1, 2)
        col_min, _ = matrix.min(dim=1, keepdim=True)
        col_min = col_min.transpose(1, 2)
        col_stats = torch.cat([col_mean, col_min], dim=2)  # [batch, col_size, 2]
        col_emb = self.col_stats_proj(col_stats)

        return row_emb, col_emb
