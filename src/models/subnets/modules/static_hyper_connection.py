"""Static Hyper-Network connections."""

import torch
from torch import nn


class StaticHyperConnection(nn.Module):
    """
    Hyper-connection with static width/depth expansion.
    """

    def __init__(self, module: nn.Module, hyper_dim: int, expansion_rate: int = 4):
        """
        Initializes the static hyper-connection.

        Args:
            module: The neural network module to wrap.
            hyper_dim: Hyper-network dimension.
            expansion_rate: Width/depth expansion rate.
        """
        super().__init__()
        self.module = module
        self.n = expansion_rate

        # Initialize slightly off-identity to preserve gradient flow at start
        self.width_mixer = nn.Parameter(torch.eye(self.n) + torch.randn(self.n, self.n) * 0.01)
        self.input_mixer = nn.Parameter(torch.randn(self.n, 1) * 0.01)
        self.depth_mixer = nn.Parameter(torch.randn(1, self.n) * 0.01)

    def forward(self, H, *args, **kwargs):
        """
        Forward pass for static hyper connection.
        """
        # H shape: (Batch, Seq, Dim, n)

        # 1. Collapse streams for the sub-layer (A_m)
        # (B, S, D, n) x (n, 1) -> (B, S, D, 1) -> (B, S, D)
        h_in = torch.matmul(H, self.input_mixer).squeeze(-1)

        # 2. Apply Sub-layer (Attention, MLP, etc.)
        y = self.module(h_in, *args, **kwargs)  # (B, S, D)

        # 3. Update Hyper Matrix
        # Width: Mix existing streams (H x A_r)
        term_width = torch.matmul(H, self.width_mixer)

        # Depth: Broadcast new info (y x B)
        term_depth = torch.matmul(y.unsqueeze(-1), self.depth_mixer)

        return term_width + term_depth
