"""GNN Layer for NARGNN."""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn

try:
    from torch_scatter import scatter_mean
except (ImportError, OSError):
    scatter_mean = None

try:
    from torch_geometric.nn import BatchNorm
except (ImportError, OSError):
    BatchNorm = None


class GNNLayer(nn.Module):
    """Simplified GNN layer for NARGNN."""

    def __init__(self, embed_dim: int, act_fn: str = "silu", agg_fn: str = "mean"):
        """Initialize Class."""
        super().__init__()
        self._check_dependencies()

        self.embed_dim = embed_dim
        self.act_fn = getattr(nn.functional, act_fn)

        self.v_lin1 = nn.Linear(embed_dim, embed_dim)
        self.v_lin2 = nn.Linear(embed_dim, embed_dim)
        self.v_lin3 = nn.Linear(embed_dim, embed_dim)
        self.v_lin4 = nn.Linear(embed_dim, embed_dim)
        self.v_bn = BatchNorm(embed_dim)  # type: ignore[calling-non-callable]

        self.e_lin = nn.Linear(embed_dim, embed_dim)
        self.e_bn = BatchNorm(embed_dim)  # type: ignore[calling-non-callable]

        self.agg_fn = agg_fn

    def _check_dependencies(self):
        """Check if required PyG dependencies are available."""
        if BatchNorm is None or scatter_mean is None:
            raise ImportError(
                "GNNLayer requires torch-geometric and torch-scatter with matching CUDA extensions. "
                "Please ensure PyG is installed and working: https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html"
            )

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        self._check_dependencies()
        x0, w0 = x, edge_attr

        x1 = self.v_lin1(x0)
        x2 = self.v_lin2(x0)
        x3 = self.v_lin3(x0)
        x4 = self.v_lin4(x0)

        edge_weighted = torch.sigmoid(w0) * x2[edge_index[1]]
        if self.agg_fn == "mean":
            aggregated = scatter_mean(edge_weighted, edge_index[0], dim=0, dim_size=x0.size(0))  # type: ignore[calling-non-callable]
        else:
            raise NotImplementedError(f"Aggregation {self.agg_fn} not implemented")

        x = x0 + self.act_fn(self.v_bn(x1 + aggregated))

        w1 = self.e_lin(w0)
        w = w0 + self.act_fn(self.e_bn(w1 + x3[edge_index[0]] + x4[edge_index[1]]))

        return x, w
