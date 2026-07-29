"""Simplified edge embedding for NARGNN."""

from __future__ import annotations

from typing import Optional

import torch
from tensordict import TensorDict
from torch import nn

try:
    from torch_geometric.data import Batch, Data
except ImportError:
    Batch = Data = None  # type: ignore


class SimplifiedEdgeEmbedding(nn.Module):
    """Simplified edge embedding for NARGNN using distance matrix."""

    def __init__(self, embed_dim: int, k_sparse: Optional[int] = None, linear_bias: bool = True):
        """Initialize Class.

        Args:
            embed_dim (int): Description of embed_dim.
            k_sparse (Optional[int]): Description of k_sparse.
            linear_bias (bool): Description of linear_bias.
        """
        super().__init__()
        assert Batch is not None, "torch_geometric required for NARGNN"

        if k_sparse is None:
            self._get_k_sparse = lambda n: max(n // 5, 10)
        elif isinstance(k_sparse, int):
            self._get_k_sparse = lambda n: k_sparse
        else:
            raise ValueError("k_sparse must be an int or None")

        self.k_sparse = k_sparse
        self.edge_embed = nn.Linear(1, embed_dim, bias=linear_bias)

    def forward(self, td: TensorDict, init_embeddings: torch.Tensor) -> Batch:  # type: ignore
        """Forward pass."""
        locs = td["locs"]
        batch_size = locs.shape[0]
        dist_matrix = torch.cdist(locs, locs, p=2)

        graph_data = []
        for i in range(batch_size):
            k = self._get_k_sparse(locs.shape[1])
            _, topk_indices = torch.topk(dist_matrix[i], k + 1, dim=-1, largest=False)

            edge_list = []
            edge_dists = []
            for node_idx in range(locs.shape[1]):
                for neighbor_idx in topk_indices[node_idx]:
                    if neighbor_idx != node_idx:
                        edge_list.append([node_idx, neighbor_idx.item()])
                        edge_dists.append(dist_matrix[i, node_idx, neighbor_idx].item())

            edge_index = torch.tensor(edge_list, dtype=torch.long, device=locs.device).t()
            edge_attr = torch.tensor(edge_dists, dtype=locs.dtype, device=locs.device).unsqueeze(-1)

            graph = Data(x=init_embeddings[i], edge_index=edge_index, edge_attr=edge_attr)  # type: ignore
            graph_data.append(graph)

        batch = Batch.from_data_list(graph_data)  # type: ignore
        batch.edge_attr = self.edge_embed(batch.edge_attr)  # type: ignore
        return batch
