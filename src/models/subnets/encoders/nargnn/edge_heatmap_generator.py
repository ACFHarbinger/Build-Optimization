"""Heatmap generator for NARGNN."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn

try:
    from torch_geometric.data import Batch
except ImportError:
    Batch = None  # type: ignore


class EdgeHeatmapGenerator(nn.Module):
    """
    MLP for converting edge embeddings to heatmaps.
    """

    def __init__(
        self,
        embed_dim: int,
        num_layers: int,
        act_fn: str | Callable = "silu",
        linear_bias: bool = True,
        undirected_graph: bool = True,
    ) -> None:
        """Initialize Class.

        Args:
            embed_dim (int): Description of embed_dim.
            num_layers (int): Description of num_layers.
            act_fn (str | Callable): Description of act_fn.
            linear_bias (bool): Description of linear_bias.
            undirected_graph (bool): Description of undirected_graph.
        """
        super().__init__()

        self.linears = nn.ModuleList([nn.Linear(embed_dim, embed_dim, bias=linear_bias) for _ in range(num_layers - 1)])
        self.output = nn.Linear(embed_dim, 1, bias=linear_bias)

        self.act = getattr(nn.functional, act_fn) if isinstance(act_fn, str) else act_fn
        self.undirected_graph = undirected_graph

    def forward(self, graph: Batch) -> torch.Tensor:  # type: ignore
        """Forward pass."""
        edge_attr = graph.edge_attr  # type: ignore
        for layer in self.linears:
            edge_attr = self.act(layer(edge_attr))
        graph.edge_attr = torch.sigmoid(self.output(edge_attr))  # type: ignore

        heatmap_logits = self._make_heatmap_logits(graph)
        return heatmap_logits

    def _make_heatmap_logits(self, batch_graph: Batch) -> torch.Tensor:  # type: ignore
        """Helper to create heatmap logits."""
        graphs = batch_graph.to_data_list() if batch_graph is not None else []
        device = graphs[0].edge_attr.device
        batch_size = len(graphs)
        num_nodes = graphs[0].x.shape[0]

        heatmap = torch.zeros(
            (batch_size, num_nodes, num_nodes),
            device=device,
            dtype=graphs[0].edge_attr.dtype,
        )

        for index, graph in enumerate(graphs):
            edge_index, edge_attr = graph.edge_index, graph.edge_attr
            heatmap[index, edge_index[0], edge_index[1]] = edge_attr.flatten()

        if heatmap.dtype in (torch.float32, torch.bfloat16):
            small_value = 1e-12
        elif heatmap.dtype == torch.float16:
            small_value = 3e-8
        else:
            raise ValueError(f"Unsupported dtype: {heatmap.dtype}")

        heatmap += small_value
        heatmap_logits = torch.log(heatmap)

        return heatmap_logits
