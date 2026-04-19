"""
Edge embedding modules for graph-based encoders.

Transforms raw edge features (e.g., distances between nodes) into learned
embeddings suitable for GNN-based encoders. Supports sparsification to
focus on k-nearest neighbors, which is critical for scaling to large instances.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import List

import torch
from torch import nn
from torch_geometric.data import Batch, Data
from torch_geometric.data.data import BaseData

from logic.src.utils.ops import get_distance_matrix, get_full_graph_edge_index, sparsify_graph


class EdgeEmbedding(nn.Module):
    """
    Base class for edge embeddings.

    Subclasses define how raw edge features are extracted from problem data
    and projected into the embedding space.
    """

    node_dim: int = 1

    def __init__(
        self,
        embed_dim: int,
        linear_bias: bool = True,
        sparsify: bool = True,
        k_sparse: int | Callable[[int], int] | None = None,
    ):
        """
        Args:
            embed_dim: Output embedding dimension for edge features.
            linear_bias: Whether to include bias in the edge projection.
            sparsify: Whether to create a sparse k-NN graph.
            k_sparse: Number of neighbors per node, or a callable(n) -> k.
                       Defaults to max(n // 5, 10).
        """
        assert Batch is not None, (
            "torch_geometric is required for EdgeEmbedding. Install via: pip install torch_geometric"
        )
        super().__init__()

        self._get_k_sparse: Callable[[int], int]
        if k_sparse is None:
            self._get_k_sparse = lambda n: max(n // 5, 10)
        elif isinstance(k_sparse, int):
            self._get_k_sparse = lambda n: k_sparse
        elif callable(k_sparse):
            self._get_k_sparse = k_sparse
        else:
            raise ValueError("k_sparse must be an int or a callable")

        self.sparsify = sparsify
        self.edge_embed = nn.Linear(self.node_dim, embed_dim, linear_bias)

    def forward(self, td, init_embeddings: torch.Tensor):
        """
        Build a batched PyG graph with embedded edge features.

        Args:
            td: TensorDict or dict with problem data (must contain 'locs').
            init_embeddings: Node embeddings [batch, n_nodes, embed_dim].

        Returns:
            PyG Batch with x, edge_index, and embedded edge_attr.
        """
        cost_matrix = get_distance_matrix(td["locs"])
        batch = self._cost_matrix_to_graph(cost_matrix, init_embeddings)
        return batch

    def _cost_matrix_to_graph(
        self,
        batch_cost_matrix: torch.Tensor,
        init_embeddings: torch.Tensor,
    ):
        """
        Convert batched cost matrices to a batched PyG graph.

        Args:
            batch_cost_matrix: [batch, n, n] distance/cost matrices.
            init_embeddings: [batch, n, embed_dim] node features.

        Returns:
            PyG Batch object.
        """
        k_sparse = self._get_k_sparse(batch_cost_matrix.shape[-1])
        graph_data: List[BaseData] = []

        for idx, cost_matrix in enumerate(batch_cost_matrix):
            if self.sparsify:
                edge_index, edge_attr = sparsify_graph(cost_matrix, k_sparse, self_loop=False)
            else:
                edge_index = get_full_graph_edge_index(cost_matrix.shape[0], self_loop=False).to(cost_matrix.device)
                edge_attr = cost_matrix[edge_index[0], edge_index[1]].unsqueeze(-1)

            graph = Data(x=init_embeddings[idx], edge_index=edge_index, edge_attr=edge_attr)
            graph_data.append(graph)

        batch = Batch.from_data_list(graph_data)
        batch.edge_attr = self.edge_embed(batch.edge_attr)
        return batch
