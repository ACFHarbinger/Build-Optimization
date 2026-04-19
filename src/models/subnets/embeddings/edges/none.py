"""none.py module.

Attributes:
    MODULE_VAR (Type): Description of module level variable.

Example:
    >>> import none
"""

from __future__ import annotations

from typing import List

import torch
from torch import nn
from torch_geometric.data import Batch, Data
from torch_geometric.data.data import BaseData

from logic.src.utils.ops import get_full_graph_edge_index


class NoEdgeEmbedding(nn.Module):
    """
    Dummy edge embedding for problems that don't use edge features.

    Creates a fully-connected graph with zero-valued edge embeddings.
    Useful for problems where only node features matter (e.g., scheduling).
    """

    def __init__(self, embed_dim: int, self_loop: bool = False, **kwargs):
        """
        Initialize NoEdgeEmbedding.

        Args:
            embed_dim: Embedding dimension.
            self_loop: Whether to include self-loops.
            **kwargs: Unused arguments.
        """
        assert Batch is not None, (
            "torch_geometric is required for NoEdgeEmbedding. Install via: pip install torch_geometric"
        )
        super().__init__()
        self.embed_dim = embed_dim
        self.self_loop = self_loop

    def forward(self, td, init_embeddings: torch.Tensor):
        """
        Generate dummy edge embeddings.

        Args:
           td: TensorDict.
           init_embeddings: Node embeddings.

        Returns:
            Batch with dummy edge attributes.
        """
        data_list: List[BaseData] = []
        n = init_embeddings.shape[1]
        device = init_embeddings.device
        edge_index = get_full_graph_edge_index(n, self_loop=self.self_loop).to(device)
        m = edge_index.shape[1]

        for node_embed in init_embeddings:
            data = Data(
                x=node_embed,
                edge_index=edge_index,
                edge_attr=torch.zeros((m, self.embed_dim), device=device),
            )
            data_list.append(data)

        return Batch.from_data_list(data_list)
